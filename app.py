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
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
            INSERT INTO runs(run_id, fingerprint, response)
            VALUES (?, ?, ?)
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
# BASIC TYPES
# ============================================================

def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def positive_safe_integer(value):
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
        text = value

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


# ============================================================
# DIGEST
# ============================================================

def make_dataset_digest(
    train_row_ids,
    eval_row_ids,
    feature_names
):
    # Exact required key order.
    obj = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names
    }

    encoded = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# ============================================================
# REQUEST FINGERPRINT
# ============================================================

def make_fingerprint(data):
    # Used only to determine whether the same runId received
    # identical selection input.
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature(feature):
    if not isinstance(feature, dict):
        return False

    # The value is arbitrary DATA.
    # Do not impose a type restriction on it.
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

def validate_selection_row(row):
    if not isinstance(row, dict):
        return False

    required = [
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    ]

    for key in required:
        if key not in row:
            return False

    # ID
    if not isinstance(row["id"], str):
        return False

    if row["id"] == "":
        return False

    # Entity
    if not isinstance(row["entity"], str):
        return False

    if row["entity"] == "":
        return False

    # Event timestamp
    event_time = parse_timestamp(
        row["eventTime"]
    )

    if event_time is None:
        return False

    # Prediction timestamp
    prediction_time = parse_timestamp(
        row["predictionTime"]
    )

    if prediction_time is None:
        return False

    # Point-in-time row sanity.
    if event_time > prediction_time:
        return False

    # Version
    if not safe_integer(
        row["version"]
    ):
        return False

    # Split
    if row["split"] not in (
        "TRAIN",
        "EVAL"
    ):
        return False

    # Features
    if not isinstance(
        row["features"],
        dict
    ):
        return False

    for feature_name, feature in row["features"].items():

        if not isinstance(
            feature_name,
            str
        ):
            return False

        if feature_name == "":
            return False

        if not validate_feature(
            feature
        ):
            return False

    return True


def validate_selection_rows(rows):
    if not isinstance(rows, list):
        return False

    if len(rows) == 0:
        return False

    ids = set()

    for row in rows:

        if not validate_selection_row(row):
            return False

        if row["id"] in ids:
            return False

        ids.add(row["id"])

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_rows(rows):
    """
    Deduplicate by:

        [entity, UTC(eventTime)]

    Winner:

        highest version

    Tie:

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

        existing = groups.get(key)

        if existing is None:
            groups[key] = row
            continue

        if row["version"] > existing["version"]:
            groups[key] = row
            continue

        if row["version"] == existing["version"]:

            if utf8_key(row["id"]) < utf8_key(
                existing["id"]
            ):
                groups[key] = row

    return list(groups.values())


# ============================================================
# SHARED + POINT-IN-TIME FEATURES
# ============================================================

def eligible_feature_names(
    retained_rows,
    forbidden_features
):
    if not retained_rows:
        return []

    forbidden = set(
        forbidden_features
    )

    # Feature must appear in EVERY retained row.
    shared = set(
        retained_rows[0]["features"].keys()
    )

    for row in retained_rows[1:]:
        shared.intersection_update(
            row["features"].keys()
        )

    # Forbidden features are excluded.
    shared.difference_update(
        forbidden
    )

    result = []

    for feature_name in shared:

        feature_is_eligible = True

        for row in retained_rows:

            feature = row["features"].get(
                feature_name
            )

            # Defensive validation.
            if not validate_feature(feature):
                feature_is_eligible = False
                break

            available_at = parse_timestamp(
                feature["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # No future information may enter the model.
            if available_at > prediction_time:
                feature_is_eligible = False
                break

        if feature_is_eligible:
            result.append(feature_name)

    return utf8_sorted(result)


# ============================================================
# TRIAL VALIDATION
# ============================================================

def validate_trial(trial):
    if not isinstance(trial, dict):
        return False

    if "trialId" not in trial:
        return False

    if "status" not in trial:
        return False

    if "evalMetric" not in trial:
        return False

    if not safe_integer(
        trial["trialId"]
    ):
        return False

    if trial["status"] not in (
        "SUCCEEDED",
        "FAILED"
    ):
        return False

    # evalMetric itself may be non-finite for a FAILED trial,
    # because only finite SUCCEEDED trials are eligible.
    return True


def validate_trials(trials):
    if not isinstance(trials, list):
        return False

    ids = set()

    for trial in trials:

        if not validate_trial(trial):
            return False

        trial_id = trial["trialId"]

        if trial_id in ids:
            return False

        ids.add(trial_id)

    return True


# ============================================================
# TRIAL SELECTION
# ============================================================

def select_best_trial(trials):

    candidates = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        if not finite_number(
            trial["evalMetric"]
        ):
            continue

        candidates.append(trial)

    if not candidates:
        return None

    # Highest metric.
    # Exact metric tie -> smallest integer trialId.
    candidates.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    return candidates[0]


# ============================================================
# SELECT REQUEST VALIDATION
# ============================================================

def validate_select_request(data):

    if not isinstance(data, dict):
        return False

    if data.get("phase") != "select":
        return False

    run_id = data.get("runId")

    if not isinstance(run_id, str):
        return False

    if run_id == "":
        return False

    if len(run_id) > 128:
        return False

    forbidden = data.get(
        "forbiddenFeatures"
    )

    if not isinstance(forbidden, list):
        return False

    for feature_name in forbidden:

        if not isinstance(
            feature_name,
            str
        ):
            return False

    if not positive_safe_integer(
        data.get("numTrialsLimit")
    ):
        return False

    if not validate_selection_rows(
        data.get("rows")
    ):
        return False

    if not validate_trials(
        data.get("trials")
    ):
        return False

    return True


# ============================================================
# SELECT
# ============================================================

def perform_selection(data):

    if not validate_select_request(data):

        run_id = ""

        if isinstance(data, dict):
            if isinstance(
                data.get("runId"),
                str
            ):
                run_id = data["runId"]

        return {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ]
        }

    # Contract failure if there are too many trials.
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
    # VALIDATE -> DEDUP
    # --------------------------------------------------------

    retained = deduplicate_rows(
        data["rows"]
    )

    # --------------------------------------------------------
    # SPLIT AFTER DEDUP
    # --------------------------------------------------------

    train_row_ids = utf8_sorted([
        row["id"]
        for row in retained
        if row["split"] == "TRAIN"
    ])

    eval_row_ids = utf8_sorted([
        row["id"]
        for row in retained
        if row["split"] == "EVAL"
    ])

    # --------------------------------------------------------
    # SHARED + POINT-IN-TIME FEATURES
    # --------------------------------------------------------

    feature_names = eligible_feature_names(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # TRIAL SELECTION
    # --------------------------------------------------------

    selected = select_best_trial(
        data["trials"]
    )

    if selected is None:

        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": [
                "NO_SUCCESSFUL_TRIAL"
            ]
        }

    # --------------------------------------------------------
    # FREEZE DATASET
    # --------------------------------------------------------

    digest = make_dataset_digest(
        train_row_ids,
        eval_row_ids,
        feature_names
    )

    return {
        "runId": data["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": []
    }


# ============================================================
# TEST ROW VALIDATION
# ============================================================

def validate_test_row(row):

    if not isinstance(row, dict):
        return False

    if "label" not in row:
        return False

    if "prediction" not in row:
        return False

    if "slice" not in row:
        return False

    label = row["label"]
    prediction = row["prediction"]

    # Binary integers only.
    # bool must NOT count as integer.
    if (
        not isinstance(label, int)
        or isinstance(label, bool)
        or label not in (0, 1)
    ):
        return False

    if (
        not isinstance(prediction, int)
        or isinstance(prediction, bool)
        or prediction not in (0, 1)
    ):
        return False

    if not isinstance(
        row["slice"],
        str
    ):
        return False

    if row["slice"] == "":
        return False

    return True


# ============================================================
# EVALUATE REQUEST VALIDATION
# ============================================================

def validate_evaluate_request(data):

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
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return False

    if not finite_number(
        data.get("metricFloor")
    ):
        return False

    if not (
        0 <= float(data["metricFloor"]) <= 1
    ):
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

        if not (
            0 <= float(floor) <= 1
        ):
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
# EVALUATE
# ============================================================

def perform_evaluation(data):

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    if not validate_evaluate_request(data):

        run_id = ""
        selected_trial_id = None
        digest = None
        bytes_processed = 0

        if isinstance(data, dict):

            if isinstance(
                data.get("runId"),
                str
            ):
                run_id = data["runId"]

            if safe_integer(
                data.get("selectedTrialId")
            ):
                selected_trial_id = data[
                    "selectedTrialId"
                ]

            if isinstance(
                data.get("datasetDigest"),
                str
            ):
                digest = data[
                    "datasetDigest"
                ]

            if safe_integer(
                data.get("bytesProcessed")
            ):
                bytes_processed = data[
                    "bytesProcessed"
                ]

        return {
            "runId": run_id,
            "selectedTrialId": selected_trial_id,
            "datasetDigest": digest,
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed": bytes_processed,
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

    lineage_valid = False

    if stored is not None:

        frozen = stored["response"]

        if (
            isinstance(frozen, dict)
            and frozen.get("reasonCodes") == []
            and safe_integer(
                frozen.get("selectedTrialId")
            )
            and isinstance(
                frozen.get("datasetDigest"),
                str
            )
            and DIGEST_RE.fullmatch(
                frozen["datasetDigest"]
            ) is not None
            and frozen["selectedTrialId"]
                == data["selectedTrialId"]
            and frozen["datasetDigest"]
                == data["datasetDigest"]
        ):
            lineage_valid = True

    if not lineage_valid:
        reasons.append(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # TEST ROW VALIDATION
    # --------------------------------------------------------

    rows = data["rows"]

    invalid_test_row = False

    for row in rows:

        if not validate_test_row(row):
            invalid_test_row = True
            break

    if invalid_test_row:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # EMPTY OR INVALID TEST DATA
    # --------------------------------------------------------

    if len(rows) == 0 or invalid_test_row:

        test_metric = None
        critical_slice_pass = False

    else:

        # ----------------------------------------------------
        # AGGREGATE
        # ----------------------------------------------------

        correct = sum(
            1
            for row in rows
            if row["label"] == row["prediction"]
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

        # ----------------------------------------------------
        # SLICES
        # ----------------------------------------------------

        slices = {}

        for row in rows:

            slice_name = row["slice"]

            if slice_name not in slices:
                slices[slice_name] = []

            slices[slice_name].append(row)

        critical_slice_pass = True

        # Required slices sorted by UTF-8 bytes.
        for name in utf8_sorted(
            list(
                data["requiredSlices"].keys()
            )
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
    # criticalSlicePass
    #
    # Only these things affect this boolean:
    # - valid lineage
    # - valid/non-empty test rows
    # - required slice existence
    # - required slice floors
    #
    # It does NOT summarize aggregate or byte gates.
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

    if (
        data["bytesProcessed"]
        > data["maxBytes"]
    ):
        reasons.append(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = "admit"

    if not lineage_valid:
        decision = "reject"

    if len(rows) == 0:
        decision = "reject"

    if invalid_test_row:
        decision = "reject"

    if test_metric is None:
        decision = "reject"

    elif (
        test_metric
        < float(data["metricFloor"])
    ):
        decision = "reject"

    if not critical_slice_pass:
        decision = "reject"

    if (
        data["bytesProcessed"]
        > data["maxBytes"]
    ):
        decision = "reject"

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
# POST /bqml
# ============================================================

@app.route(
    "/bqml",
    methods=["POST"]
)
def bqml():

    try:

        # Must accept application/json.
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
        if phase not in (
            "select",
            "evaluate"
        ):
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        # ====================================================
        # SELECT
        # ====================================================

        if phase == "select":

            run_id = data.get("runId")

            valid_run_id = (
                isinstance(run_id, str)
                and run_id != ""
                and len(run_id) <= 128
            )

            # Malformed selection has the normal selection
            # response shape and is not persisted unless it has
            # a valid runId.
            if not valid_run_id:
                return jsonify(
                    perform_selection(data)
                )

            fingerprint = make_fingerprint(
                data
            )

            previous = load_run(
                run_id
            )

            # ------------------------------------------------
            # IDENTICAL REPLAY
            # ------------------------------------------------

            if previous is not None:

                if (
                    previous["fingerprint"]
                    == fingerprint
                ):
                    return jsonify(
                        previous["response"]
                    )

                # Same runId but different selection input.
                return jsonify({
                    "error": "RUN_ID_CONFLICT"
                }), 409

            # ------------------------------------------------
            # NEW SELECTION
            # ------------------------------------------------

            response = perform_selection(
                data
            )

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
