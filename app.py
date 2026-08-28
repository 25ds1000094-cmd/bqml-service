from flask import Flask, request, jsonify
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile

app = Flask(__name__)

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(tempfile.gettempdir(), "bqml_state.sqlite3")
)

MAX_SAFE_INTEGER = 9007199254740991

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            response TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_run(run_id):
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT fingerprint, response
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "fingerprint": row[0],
        "response": json.loads(row[1])
    }


def save_run(run_id, fingerprint, response):
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO runs
                (run_id, fingerprint, response)
            VALUES
                (?, ?, ?)
            """,
            (
                run_id,
                fingerprint,
                json.dumps(
                    response,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            )
        )
        conn.commit()
    finally:
        conn.close()


# ============================================================
# BASIC VALIDATION
# ============================================================

def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_SAFE_INTEGER
    )


def finite_number(value):
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def utf8_key(value):
    return value.encode("utf-8")


def utf8_sorted(values):
    return sorted(values, key=utf8_key)


def sorted_codes(codes):
    return sorted(set(codes), key=utf8_key)


# ============================================================
# TIMESTAMPS
# ============================================================

def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    if TIMESTAMP_RE.fullmatch(value) is None:
        return None

    try:
        normalized = value

        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        dt = datetime.fromisoformat(normalized)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


# ============================================================
# DIGEST
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def dataset_digest(train_ids, eval_ids, feature_names):
    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    return hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()


def request_fingerprint(data):
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# ============================================================
# SELECTION FEATURE VALIDATION
# ============================================================

def valid_feature(feature):
    if not isinstance(feature, dict):
        return False

    if "value" not in feature:
        return False

    if "availableAt" not in feature:
        return False

    if parse_timestamp(feature["availableAt"]) is None:
        return False

    return True


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

def valid_selection_row(row):
    if not isinstance(row, dict):
        return False

    required = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    }

    if set(row.keys()) != required:
        return False

    if not isinstance(row["id"], str):
        return False

    if row["id"] == "":
        return False

    if not isinstance(row["entity"], str):
        return False

    if row["entity"] == "":
        return False

    if parse_timestamp(row["eventTime"]) is None:
        return False

    if parse_timestamp(row["predictionTime"]) is None:
        return False

    if not safe_integer(row["version"]):
        return False

    if row["split"] not in ("TRAIN", "EVAL"):
        return False

    if not isinstance(row["features"], dict):
        return False

    for feature_name, feature in row["features"].items():

        if not isinstance(feature_name, str):
            return False

        if feature_name == "":
            return False

        if not valid_feature(feature):
            return False

    return True


def valid_selection_rows(rows):
    if not isinstance(rows, list):
        return False

    if len(rows) == 0:
        return False

    ids = set()

    for row in rows:

        if not valid_selection_row(row):
            return False

        if row["id"] in ids:
            return False

        ids.add(row["id"])

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(rows):
    """
    Deduplicate by [entity, UTC(eventTime)].

    Winner:
      1. Highest version
      2. UTF-8-byte-smallest ID on equal version
    """

    groups = {}

    for row in rows:

        event_utc = parse_timestamp(
            row["eventTime"]
        )

        key = (
            row["entity"],
            event_utc
        )

        current = groups.get(key)

        if current is None:
            groups[key] = row
            continue

        if row["version"] > current["version"]:
            groups[key] = row
            continue

        if row["version"] == current["version"]:
            if utf8_key(row["id"]) < utf8_key(current["id"]):
                groups[key] = row

    return list(groups.values())


# ============================================================
# POINT-IN-TIME / SHARED FEATURES
# ============================================================

def eligible_features(retained_rows, forbidden_features):
    if not retained_rows:
        return []

    forbidden = set(forbidden_features)

    # Feature must exist in EVERY retained row.
    shared = set(
        retained_rows[0]["features"].keys()
    )

    for row in retained_rows[1:]:
        shared.intersection_update(
            row["features"].keys()
        )

    # Forbidden features are never eligible.
    shared.difference_update(forbidden)

    eligible = []

    for feature_name in shared:

        feature_is_valid = True

        for row in retained_rows:

            available_at = parse_timestamp(
                row["features"][feature_name]["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            if available_at is None or prediction_time is None:
                feature_is_valid = False
                break

            # Point-in-time rule:
            # feature must have been available by prediction time.
            if available_at > prediction_time:
                feature_is_valid = False
                break

        if feature_is_valid:
            eligible.append(feature_name)

    return utf8_sorted(eligible)


# ============================================================
# TRIAL VALIDATION
# ============================================================

def valid_trial(trial):
    if not isinstance(trial, dict):
        return False

    required = {
        "trialId",
        "status",
        "evalMetric"
    }

    if set(trial.keys()) != required:
        return False

    if not safe_integer(trial["trialId"]):
        return False

    if trial["status"] not in (
        "SUCCEEDED",
        "FAILED"
    ):
        return False

    return True


def valid_trials(trials):
    if not isinstance(trials, list):
        return False

    ids = set()

    for trial in trials:

        if not valid_trial(trial):
            return False

        if trial["trialId"] in ids:
            return False

        ids.add(trial["trialId"])

    return True


# ============================================================
# TRIAL SELECTION
# ============================================================

def select_trial(trials):
    eligible = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        # Only finite SUCCEEDED trials are eligible.
        if not finite_number(trial["evalMetric"]):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest metric first.
    # Exact tie -> smallest integer trialId.
    eligible.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    return eligible[0]


# ============================================================
# SELECTION INPUT VALIDATION
# ============================================================

def valid_selection(data):
    if not isinstance(data, dict):
        return False

    required = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials"
    }

    if set(data.keys()) != required:
        return False

    if data["phase"] != "select":
        return False

    run_id = data["runId"]

    if not isinstance(run_id, str):
        return False

    if run_id == "" or len(run_id) > 128:
        return False

    if not isinstance(data["forbiddenFeatures"], list):
        return False

    if not all(
        isinstance(feature, str)
        for feature in data["forbiddenFeatures"]
    ):
        return False

    if not positive_integer(
        data["numTrialsLimit"]
    ):
        return False

    if not valid_selection_rows(
        data["rows"]
    ):
        return False

    if not valid_trials(
        data["trials"]
    ):
        return False

    return True


# ============================================================
# SELECTION
# ============================================================

def perform_selection(data):

    if not valid_selection(data):

        return {
            "runId": (
                data.get("runId", "")
                if isinstance(data, dict)
                else ""
            ),
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ]
        }

    # More trials than allowed is a contract failure.
    if len(data["trials"]) > data["numTrialsLimit"]:

        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": [
                "TRIAL_LIMIT_EXCEEDED"
            ]
        }

    # --------------------------------------------------------
    # FIRST: DEDUPLICATE
    # --------------------------------------------------------

    retained = deduplicate(data["rows"])

    # --------------------------------------------------------
    # SECOND: SPLIT
    # --------------------------------------------------------

    train_ids = utf8_sorted([
        row["id"]
        for row in retained
        if row["split"] == "TRAIN"
    ])

    eval_ids = utf8_sorted([
        row["id"]
        for row in retained
        if row["split"] == "EVAL"
    ])

    # --------------------------------------------------------
    # THIRD: SHARED + POINT-IN-TIME-SAFE FEATURES
    # --------------------------------------------------------

    feature_names = eligible_features(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # FOURTH: TRIAL SELECTION
    # --------------------------------------------------------

    selected = select_trial(
        data["trials"]
    )

    if selected is None:

        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": [
                "NO_SUCCESSFUL_TRIAL"
            ]
        }

    # --------------------------------------------------------
    # DIGEST
    # --------------------------------------------------------

    digest = dataset_digest(
        train_ids,
        eval_ids,
        feature_names
    )

    return {
        "runId": data["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": []
    }


# ============================================================
# FINAL TEST ROW VALIDATION
# ============================================================

def valid_test_row(row):
    if not isinstance(row, dict):
        return False

    required = {
        "label",
        "prediction",
        "slice"
    }

    if set(row.keys()) != required:
        return False

    if (
        not isinstance(row["label"], int)
        or isinstance(row["label"], bool)
        or row["label"] not in (0, 1)
    ):
        return False

    if (
        not isinstance(row["prediction"], int)
        or isinstance(row["prediction"], bool)
        or row["prediction"] not in (0, 1)
    ):
        return False

    if not isinstance(row["slice"], str):
        return False

    if row["slice"] == "":
        return False

    return True


# ============================================================
# EVALUATION INPUT VALIDATION
# ============================================================

def valid_evaluation(data):
    if not isinstance(data, dict):
        return False

    required = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes"
    }

    if set(data.keys()) != required:
        return False

    if data["phase"] != "evaluate":
        return False

    if not isinstance(data["runId"], str):
        return False

    if data["runId"] == "" or len(data["runId"]) > 128:
        return False

    if not safe_integer(data["selectedTrialId"]):
        return False

    if (
        not isinstance(data["datasetDigest"], str)
        or DIGEST_RE.fullmatch(data["datasetDigest"]) is None
    ):
        return False

    if not finite_number(data["metricFloor"]):
        return False

    if not 0 <= float(data["metricFloor"]) <= 1:
        return False

    required_slices = data["requiredSlices"]

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():

        if not isinstance(name, str):
            return False

        if name == "":
            return False

        if not finite_number(floor):
            return False

        if not 0 <= float(floor) <= 1:
            return False

    if not isinstance(data["rows"], list):
        return False

    if not safe_integer(data["bytesProcessed"]):
        return False

    if not safe_integer(data["maxBytes"]):
        return False

    return True


# ============================================================
# ROUNDING
# ============================================================

def round12(value):
    return float(
        Decimal(str(value)).quantize(
            Decimal("0.000000000001"),
            rounding=ROUND_HALF_UP
        )
    )


# ============================================================
# EVALUATION
# ============================================================

def perform_evaluation(data):

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    if not valid_evaluation(data):

        return {
            "runId": (
                data.get("runId", "")
                if isinstance(data, dict)
                else ""
            ),
            "selectedTrialId": (
                data.get("selectedTrialId")
                if (
                    isinstance(data, dict)
                    and safe_integer(
                        data.get("selectedTrialId")
                    )
                )
                else None
            ),
            "datasetDigest": (
                data.get("datasetDigest")
                if isinstance(data, dict)
                else None
            ),
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed": (
                data.get("bytesProcessed")
                if (
                    isinstance(data, dict)
                    and safe_integer(
                        data.get("bytesProcessed")
                    )
                )
                else 0
            ),
            "reasonCodes": [
                "INVALID_INPUT"
            ]
        }

    reasons = []

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    stored = load_run(data["runId"])

    lineage_valid = False

    if stored is not None:

        saved = stored["response"]

        if (
            saved.get("reasonCodes") == []
            and saved.get("selectedTrialId") is not None
            and saved.get("datasetDigest") is not None
            and saved.get("selectedTrialId")
                == data["selectedTrialId"]
            and saved.get("datasetDigest")
                == data["datasetDigest"]
        ):
            lineage_valid = True

    if not lineage_valid:
        reasons.append("INVALID_LINEAGE")

    # --------------------------------------------------------
    # FINAL TEST ROWS
    # --------------------------------------------------------

    rows = data["rows"]

    invalid_test_row = any(
        not valid_test_row(row)
        for row in rows
    )

    if invalid_test_row:
        reasons.append("INVALID_TEST_ROW")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if len(rows) == 0 or invalid_test_row:

        # Required by spec:
        # empty/invalid test data => null metric.
        test_metric = None
        critical_slice_pass = False

    else:

        # ----------------------------------------------------
        # AGGREGATE ACCURACY
        # ----------------------------------------------------

        correct = sum(
            1
            for row in rows
            if row["label"] == row["prediction"]
        )

        test_metric = round12(
            correct / len(rows)
        )

        if test_metric < float(
            data["metricFloor"]
        ):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # REQUIRED SLICES
        # ----------------------------------------------------

        slices = {}

        for row in rows:
            slices.setdefault(
                row["slice"],
                []
            ).append(row)

        # This represents ONLY the required-slice gates.
        critical_slice_pass = True

        for name in utf8_sorted(
            list(data["requiredSlices"].keys())
        ):

            if name not in slices:

                reasons.append(
                    "MISSING_SLICE:" + name
                )

                critical_slice_pass = False
                continue

            slice_rows = slices[name]

            slice_correct = sum(
                1
                for row in slice_rows
                if row["label"] == row["prediction"]
            )

            slice_metric = round12(
                slice_correct / len(slice_rows)
            )

            floor = float(
                data["requiredSlices"][name]
            )

            if slice_metric < floor:

                reasons.append(
                    "SLICE_FLOOR:" + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # CRITICAL SLICE FLAG
    # --------------------------------------------------------

    if not lineage_valid:
        critical_slice_pass = False

    if invalid_test_row:
        critical_slice_pass = False

    if len(rows) == 0:
        critical_slice_pass = False

    # --------------------------------------------------------
    # BYTE LIMIT
    # --------------------------------------------------------

    byte_pass = (
        data["bytesProcessed"]
        <= data["maxBytes"]
    )

    if not byte_pass:
        reasons.append("BYTE_LIMIT")

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    rows_pass = (
        len(rows) > 0
        and not invalid_test_row
    )

    aggregate_pass = (
        test_metric is not None
        and test_metric >= float(
            data["metricFloor"]
        )
    )

    decision = "admit"

    if not lineage_valid:
        decision = "reject"

    if not rows_pass:
        decision = "reject"

    if not aggregate_pass:
        decision = "reject"

    if not critical_slice_pass:
        decision = "reject"

    if not byte_pass:
        decision = "reject"

    # --------------------------------------------------------
    # EXACT OUTPUT SHAPE
    # --------------------------------------------------------

    return {
        "runId": data["runId"],
        "selectedTrialId": data["selectedTrialId"],
        "datasetDigest": data["datasetDigest"],
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": data["bytesProcessed"],
        "reasonCodes": sorted_codes(reasons)
    }


# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.route("/bqml", methods=["POST"])
def bqml():

    try:

        if not request.is_json:
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        phase = data.get("phase")

        # Unknown or missing phase.
        if phase not in ("select", "evaluate"):
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        # ====================================================
        # SELECT
        # ====================================================

        if phase == "select":

            run_id = data.get("runId")

            # Invalid runId cannot be persisted.
            if not (
                isinstance(run_id, str)
                and run_id != ""
                and len(run_id) <= 128
            ):
                return jsonify(
                    perform_selection(data)
                )

            fingerprint = request_fingerprint(data)

            previous = load_run(run_id)

            # ------------------------------------------------
            # IDENTICAL REPLAY
            # ------------------------------------------------

            if previous is not None:

                if previous["fingerprint"] == fingerprint:
                    return jsonify(
                        previous["response"]
                    )

                # Same runId with different selection input.
                return jsonify({
                    "error": "RUN_ID_CONFLICT"
                }), 409

            # ------------------------------------------------
            # NEW SELECTION
            # ------------------------------------------------

            response = perform_selection(data)

            save_run(
                run_id,
                fingerprint,
                response
            )

            return jsonify(response)

        # ====================================================
        # EVALUATE
        # ====================================================

        return jsonify(
            perform_evaluation(data)
        )

    except Exception:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400


# ============================================================
# ROOT
# ============================================================

@app.route("/")
def home():
    return "BQML service running"


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8080"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
