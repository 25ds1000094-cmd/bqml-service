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
    conn = sqlite3.connect(
        DB_PATH,
        timeout=10
    )

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
    serialized = json.dumps(
        response,
        ensure_ascii=False,
        separators=(",", ":")
    )

    conn = get_db()

    try:
        conn.execute(
            """
            INSERT INTO runs
            (run_id, fingerprint, response)
            VALUES (?, ?, ?)
            """,
            (
                run_id,
                fingerprint,
                serialized
            )
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        conn.rollback()
        return False

    finally:
        conn.close()


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_integer(value):
    return (
        type(value) is int
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def positive_integer(value):
    return (
        type(value) is int
        and value > 0
    )


def finite_number(value):
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def utf8_key(value):
    return value.encode("utf-8")


def utf8_sorted(values):
    return sorted(values, key=utf8_key)


def sorted_codes(codes):
    return sorted(
        set(codes),
        key=utf8_key
    )


# ============================================================
# TIMESTAMPS
# ============================================================

def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    if not TIMESTAMP_RE.fullmatch(value):
        return None

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        dt = datetime.fromisoformat(value)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except (TypeError, ValueError, OverflowError):
        return None


# ============================================================
# DIGEST
# ============================================================

def dataset_digest(
    train_ids,
    eval_ids,
    feature_names
):
    obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    encoded = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def request_fingerprint(data):
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

def valid_feature(feature):
    if not isinstance(feature, dict):
        return False

    if "value" not in feature:
        return False

    if "availableAt" not in feature:
        return False

    return parse_timestamp(
        feature["availableAt"]
    ) is not None


def valid_selection_row(row):
    if not isinstance(row, dict):
        return False

    required = (
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    )

    for key in required:
        if key not in row:
            return False

    # Row ID is data identifying the row.
    if not isinstance(row["id"], str):
        return False

    # Entity is identifying data.
    if not isinstance(row["entity"], str):
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

    for name, feature in row["features"].items():
        if not isinstance(name, str):
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
    Deduplicate by:

        [entity, UTC(eventTime)]

    Winner:

        highest version

    Exact version tie:

        UTF-8-byte-smallest ID
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

        old = groups.get(key)

        if old is None:
            groups[key] = row
            continue

        if row["version"] > old["version"]:
            groups[key] = row
            continue

        if row["version"] == old["version"]:
            if utf8_key(row["id"]) < utf8_key(old["id"]):
                groups[key] = row

    return list(groups.values())


# ============================================================
# SHARED POINT-IN-TIME FEATURES
# ============================================================

def eligible_features(
    retained_rows,
    forbidden_features
):
    if not retained_rows:
        return []

    forbidden = set(forbidden_features)

    # Start with features from the first retained row.
    shared = set(
        retained_rows[0]["features"].keys()
    )

    # Feature MUST appear in every retained row.
    for row in retained_rows[1:]:
        shared.intersection_update(
            row["features"].keys()
        )

    # Forbidden features are excluded.
    shared.difference_update(forbidden)

    result = []

    for feature_name in shared:
        ok = True

        for row in retained_rows:
            available_at = parse_timestamp(
                row["features"][feature_name]["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # Exact contract:
            #
            # availableAt <= predictionTime
            #
            # Equality is allowed.
            if available_at > prediction_time:
                ok = False
                break

        if ok:
            result.append(feature_name)

    return utf8_sorted(result)


# ============================================================
# TRIAL VALIDATION / SELECTION
# ============================================================

def valid_trial(trial):
    if not isinstance(trial, dict):
        return False

    if "trialId" not in trial:
        return False

    if "status" not in trial:
        return False

    if "evalMetric" not in trial:
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


def select_trial(trials):
    eligible = []

    for trial in trials:
        if trial["status"] != "SUCCEEDED":
            continue

        if not finite_number(
            trial["evalMetric"]
        ):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest metric.
    #
    # Exact metric tie -> smallest trialId.
    eligible.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    return eligible[0]


# ============================================================
# SELECTION INPUT
# ============================================================

def valid_selection(data):
    if not isinstance(data, dict):
        return False

    if data.get("phase") != "select":
        return False

    run_id = data.get("runId")

    if not isinstance(run_id, str):
        return False

    if run_id == "" or len(run_id) > 128:
        return False

    forbidden = data.get(
        "forbiddenFeatures"
    )

    if not isinstance(forbidden, list):
        return False

    if not all(
        isinstance(x, str)
        for x in forbidden
    ):
        return False

    if not positive_integer(
        data.get("numTrialsLimit")
    ):
        return False

    if not valid_selection_rows(
        data.get("rows")
    ):
        return False

    if not valid_trials(
        data.get("trials")
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

    # Contract failure.
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
    # 1. VALIDATED ROWS
    # 2. DEDUPLICATE
    # --------------------------------------------------------

    retained = deduplicate(
        data["rows"]
    )

    # --------------------------------------------------------
    # 3. SPLIT AFTER DEDUPLICATION
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
    # 4. SHARED FEATURES + POINT-IN-TIME AVAILABILITY
    # --------------------------------------------------------

    feature_names = eligible_features(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # 5. MODEL SELECTION
    # --------------------------------------------------------
    #
    # Only selection rows are available here.
    # Evaluation/final-test rows are handled exclusively by
    # perform_evaluation().
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
    # 6. FREEZE DATASET LINEAGE
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

    # Must be REAL JSON integers.
    # bool must not pass as 0/1.
    if type(row.get("label")) is not int:
        return False

    if row["label"] not in (0, 1):
        return False

    if type(row.get("prediction")) is not int:
        return False

    if row["prediction"] not in (0, 1):
        return False

    if not isinstance(
        row.get("slice"),
        str
    ):
        return False

    if row["slice"] == "":
        return False

    return True


# ============================================================
# EVALUATION INPUT
# ============================================================

def valid_evaluation(data):
    if not isinstance(data, dict):
        return False

    if data.get("phase") != "evaluate":
        return False

    run_id = data.get("runId")

    if not isinstance(run_id, str):
        return False

    if run_id == "" or len(run_id) > 128:
        return False

    if not safe_integer(
        data.get("selectedTrialId")
    ):
        return False

    digest = data.get(
        "datasetDigest"
    )

    if (
        not isinstance(digest, str)
        or not DIGEST_RE.fullmatch(digest)
    ):
        return False

    metric_floor = data.get(
        "metricFloor"
    )

    if not finite_number(metric_floor):
        return False

    if not 0 <= float(metric_floor) <= 1:
        return False

    required_slices = data.get(
        "requiredSlices"
    )

    if not isinstance(
        required_slices,
        dict
    ):
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

    rows = data.get("rows")

    if not isinstance(rows, list):
        return False

    if not safe_integer(
        data.get("bytesProcessed")
    ):
        return False

    if not safe_integer(
        data.get("maxBytes")
    ):
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
# LINEAGE
# ============================================================

def valid_stored_success(response):
    if not isinstance(response, dict):
        return False

    if response.get("reasonCodes") != []:
        return False

    if not safe_integer(
        response.get("selectedTrialId")
    ):
        return False

    digest = response.get(
        "datasetDigest"
    )

    if (
        not isinstance(digest, str)
        or not DIGEST_RE.fullmatch(digest)
    ):
        return False

    if not isinstance(
        response.get("trainRowIds"),
        list
    ):
        return False

    if not isinstance(
        response.get("evalRowIds"),
        list
    ):
        return False

    if not isinstance(
        response.get("featureNames"),
        list
    ):
        return False

    return True


def lineage_matches(data, stored):
    if stored is None:
        return False

    response = stored.get(
        "response"
    )

    if not valid_stored_success(response):
        return False

    return (
        response["selectedTrialId"]
        == data["selectedTrialId"]
        and
        response["datasetDigest"]
        == data["datasetDigest"]
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
                data.get("bytesProcessed", 0)
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

    stored = load_run(
        data["runId"]
    )

    lineage_ok = lineage_matches(
        data,
        stored
    )

    if not lineage_ok:
        reasons.append(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # TEST ROW VALIDATION
    # --------------------------------------------------------

    rows = data["rows"]

    invalid_row = any(
        not valid_test_row(row)
        for row in rows
    )

    if invalid_row:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # EMPTY OR INVALID TEST DATA
    # --------------------------------------------------------

    if not rows or invalid_row:

        test_metric = None
        critical_slice_pass = False

    else:

        # ====================================================
        # AGGREGATE
        # ====================================================

        correct = sum(
            row["label"] == row["prediction"]
            for row in rows
        )

        test_metric = round12(
            correct / len(rows)
        )

        if (
            test_metric
            < float(data["metricFloor"])
        ):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ====================================================
        # REQUIRED SLICES
        # ====================================================

        grouped = {}

        for row in rows:
            grouped.setdefault(
                row["slice"],
                []
            ).append(row)

        critical_slice_pass = True

        for name in utf8_sorted(
            data["requiredSlices"].keys()
        ):

            if name not in grouped:

                reasons.append(
                    "MISSING_SLICE:" + name
                )

                critical_slice_pass = False
                continue

            slice_rows = grouped[name]

            slice_correct = sum(
                row["label"] == row["prediction"]
                for row in slice_rows
            )

            slice_accuracy = round12(
                slice_correct / len(slice_rows)
            )

            floor = float(
                data["requiredSlices"][name]
            )

            if slice_accuracy < floor:

                reasons.append(
                    "SLICE_FLOOR:" + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # criticalSlicePass
    # --------------------------------------------------------
    #
    # Must be false for:
    #   invalid lineage
    #   invalid test row
    #   empty rows
    #   missing required slice
    #   failed required slice floor
    #
    # It does NOT represent aggregate or byte status.
    # --------------------------------------------------------

    if not lineage_ok:
        critical_slice_pass = False

    if invalid_row:
        critical_slice_pass = False

    if not rows:
        critical_slice_pass = False

    # --------------------------------------------------------
    # BYTE GATE
    # --------------------------------------------------------

    bytes_ok = (
        data["bytesProcessed"]
        <= data["maxBytes"]
    )

    if not bytes_ok:
        reasons.append(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # AGGREGATE GATE
    # --------------------------------------------------------

    aggregate_ok = (
        test_metric is not None
        and
        test_metric
        >= float(data["metricFloor"])
    )

    # --------------------------------------------------------
    # TEST ROW GATE
    # --------------------------------------------------------

    rows_ok = (
        len(rows) > 0
        and not invalid_row
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    admit = (
        lineage_ok
        and rows_ok
        and aggregate_ok
        and critical_slice_pass
        and bytes_ok
    )

    decision = (
        "admit"
        if admit
        else "reject"
    )

    # --------------------------------------------------------
    # EXACT OUTPUT
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
# ENDPOINT
# ============================================================

@app.route(
    "/bqml",
    methods=["POST"]
)
def bqml():

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

    # Exact required response for missing/unknown phase.
    if phase not in (
        "select",
        "evaluate"
    ):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        run_id = data.get(
            "runId"
        )

        # Malformed run ID means this request is not eligible
        # for state persistence.
        if not (
            isinstance(run_id, str)
            and run_id != ""
            and len(run_id) <= 128
        ):
            return jsonify(
                perform_selection(data)
            )

        fingerprint = request_fingerprint(
            data
        )

        previous = load_run(
            run_id
        )

        # Identical replay.
        if previous is not None:

            if (
                previous["fingerprint"]
                == fingerprint
            ):
                return jsonify(
                    previous["response"]
                )

            # Same ID, different selection input.
            return jsonify({
                "error": "RUN_ID_CONFLICT"
            }), 409

        response = perform_selection(
            data
        )

        inserted = save_run(
            run_id,
            fingerprint,
            response
        )

        # Handle concurrent same-run insertion.
        if not inserted:

            previous = load_run(
                run_id
            )

            if previous is not None:

                if (
                    previous["fingerprint"]
                    == fingerprint
                ):
                    return jsonify(
                        previous["response"]
                    )

                return jsonify({
                    "error": "RUN_ID_CONFLICT"
                }), 409

            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        return jsonify(response)

    # ========================================================
    # EVALUATE
    # ========================================================

    return jsonify(
        perform_evaluation(data)
    )


# ============================================================
# ROOT
# ============================================================

@app.route("/")
def home():
    return "BQML service running"


# ============================================================
# LOCAL
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
