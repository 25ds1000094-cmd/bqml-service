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
# TIMESTAMP VALIDATION
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

def dataset_digest(
    train_row_ids,
    eval_row_ids,
    feature_names
):
    obj = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names
    }

    text = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# RUN FINGERPRINT
# ============================================================

def request_fingerprint(data):
    text = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# FEATURE VALIDATION
# ============================================================

def valid_feature(feature):
    if not isinstance(feature, dict):
        return False

    # "value" is arbitrary data.
    # Do NOT validate or interpret its contents.
    if "value" not in feature:
        return False

    if "availableAt" not in feature:
        return False

    if parse_timestamp(
        feature["availableAt"]
    ) is None:
        return False

    return True


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

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

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if not isinstance(row["id"], str):
        return False

    if row["id"] == "":
        return False

    # --------------------------------------------------------
    # ENTITY
    # --------------------------------------------------------

    if not isinstance(row["entity"], str):
        return False

    if row["entity"] == "":
        return False

    # --------------------------------------------------------
    # EVENT TIME
    # --------------------------------------------------------

    if parse_timestamp(
        row["eventTime"]
    ) is None:
        return False

    # --------------------------------------------------------
    # PREDICTION TIME
    # --------------------------------------------------------

    if parse_timestamp(
        row["predictionTime"]
    ) is None:
        return False

    # IMPORTANT:
    # Do NOT reject eventTime > predictionTime here.
    # Row validity and feature availability are separate
    # requirements in the contract.

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    if not safe_integer(
        row["version"]
    ):
        return False

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    if row["split"] not in (
        "TRAIN",
        "EVAL"
    ):
        return False

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    if not isinstance(
        row["features"],
        dict
    ):
        return False

    for name, feature in row["features"].items():

        if not isinstance(name, str):
            return False

        if name == "":
            return False

        if not valid_feature(feature):
            return False

    return True


def valid_selection_rows(rows):
    if not isinstance(rows, list):
        return False

    # Selection rows must be non-empty.
    if len(rows) == 0:
        return False

    ids = set()

    for row in rows:

        if not valid_selection_row(row):
            return False

        # IDs unique within supplied array.
        if row["id"] in ids:
            return False

        ids.add(row["id"])

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_rows(rows):
    """
    Key:
        [entity, UTC(eventTime)]

    Winner:
        highest version

    Tie:
        UTF-8-byte-smallest ID
    """

    groups = {}

    for row in rows:

        event_time_utc = parse_timestamp(
            row["eventTime"]
        )

        key = (
            row["entity"],
            event_time_utc
        )

        existing = groups.get(key)

        if existing is None:
            groups[key] = row
            continue

        # Highest version wins.
        if row["version"] > existing["version"]:
            groups[key] = row
            continue

        # Equal version -> UTF-8 smallest ID.
        if row["version"] == existing["version"]:

            if (
                utf8_key(row["id"])
                <
                utf8_key(existing["id"])
            ):
                groups[key] = row

    return list(groups.values())


# ============================================================
# SHARED FEATURE SELECTION
# ============================================================

def get_eligible_features(
    retained_rows,
    forbidden_features
):
    """
    Feature is eligible iff:

    - it appears in EVERY retained row
    - it is not forbidden
    - availableAt <= predictionTime in EVERY retained row
    """

    if not retained_rows:
        return []

    forbidden = set(
        forbidden_features
    )

    # Start with first row's features.
    shared = set(
        retained_rows[0]["features"].keys()
    )

    # Intersect with every other retained row.
    for row in retained_rows[1:]:

        shared.intersection_update(
            row["features"].keys()
        )

    # Remove forbidden features.
    shared = {
        name
        for name in shared
        if name not in forbidden
    }

    eligible = []

    for name in shared:

        feature_ok = True

        for row in retained_rows:

            feature = row["features"].get(
                name
            )

            if not valid_feature(feature):
                feature_ok = False
                break

            available_at = parse_timestamp(
                feature["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # Point-in-time leakage check.
            if available_at > prediction_time:
                feature_ok = False
                break

        if feature_ok:
            eligible.append(name)

    return utf8_sorted(eligible)


# ============================================================
# TRIAL VALIDATION
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

    if not safe_integer(
        trial["trialId"]
    ):
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

        ids.add(
            trial["trialId"]
        )

    return True


# ============================================================
# SELECT BEST TRIAL
# ============================================================

def choose_trial(trials):

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
    # Exact tie -> smallest trialId.
    eligible.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    return eligible[0]


# ============================================================
# VALIDATE SELECT REQUEST
# ============================================================

def validate_selection(data):

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
# PERFORM SELECTION
# ============================================================

def perform_selection(data):

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    if not validate_selection(data):

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

    # --------------------------------------------------------
    # TRIAL LIMIT
    # --------------------------------------------------------

    if (
        len(data["trials"])
        > data["numTrialsLimit"]
    ):

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
    # ROW VALIDATION HAS ALREADY HAPPENED.
    #
    # NOW DEDUPLICATE.
    # --------------------------------------------------------

    retained_rows = deduplicate_rows(
        data["rows"]
    )

    # --------------------------------------------------------
    # SPLIT AFTER DEDUPLICATION
    # --------------------------------------------------------

    train_row_ids = utf8_sorted([
        row["id"]
        for row in retained_rows
        if row["split"] == "TRAIN"
    ])

    eval_row_ids = utf8_sorted([
        row["id"]
        for row in retained_rows
        if row["split"] == "EVAL"
    ])

    # --------------------------------------------------------
    # SHARED FEATURES AFTER DEDUPLICATION
    # --------------------------------------------------------

    feature_names = get_eligible_features(
        retained_rows,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # TRIAL SELECTION
    # --------------------------------------------------------

    selected = choose_trial(
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
    # FREEZE DATASET DIGEST
    # --------------------------------------------------------

    digest = dataset_digest(
        train_row_ids,
        eval_row_ids,
        feature_names
    )

    # --------------------------------------------------------
    # EXACT SELECTION RESPONSE
    # --------------------------------------------------------

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
# FINAL TEST ROW VALIDATION
# ============================================================

def valid_test_row(row):

    if not isinstance(row, dict):
        return False

    if "label" not in row:
        return False

    if "prediction" not in row:
        return False

    if "slice" not in row:
        return False

    # Must be binary integers, not booleans.
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

    if not isinstance(
        row["slice"],
        str
    ):
        return False

    if row["slice"] == "":
        return False

    return True


# ============================================================
# VALIDATE EVALUATE REQUEST
# ============================================================

def validate_evaluation(data):

    if not isinstance(data, dict):
        return False

    if data.get("phase") != "evaluate":
        return False

    run_id = data.get("runId")

    if not isinstance(run_id, str):
        return False

    if run_id == "" or len(run_id) > 128:
        return False

    selected_trial_id = data.get(
        "selectedTrialId"
    )

    if not safe_integer(
        selected_trial_id
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

    metric_floor = data.get(
        "metricFloor"
    )

    if not finite_number(
        metric_floor
    ):
        return False

    if not (
        0 <= float(metric_floor) <= 1
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
# ROUND TO 12 DECIMALS
# ============================================================

def round12(value):
    return float(
        Decimal(str(value)).quantize(
            Decimal("0.000000000001"),
            rounding=ROUND_HALF_UP
        )
    )


# ============================================================
# PERFORM EVALUATION
# ============================================================

def perform_evaluation(data):

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    if not validate_evaluation(data):

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

        saved = stored["response"]

        if (
            isinstance(saved, dict)
            and saved.get("reasonCodes") == []
            and safe_integer(
                saved.get("selectedTrialId")
            )
            and isinstance(
                saved.get("datasetDigest"),
                str
            )
            and DIGEST_RE.fullmatch(
                saved["datasetDigest"]
            ) is not None
            and saved["selectedTrialId"]
                == data["selectedTrialId"]
            and saved["datasetDigest"]
                == data["datasetDigest"]
        ):
            lineage_valid = True

    if not lineage_valid:
        reasons.append(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # TEST ROWS
    # --------------------------------------------------------

    rows = data["rows"]

    invalid_row = False

    for row in rows:

        if not valid_test_row(row):
            invalid_row = True
            break

    if invalid_row:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # EMPTY OR INVALID ROWS
    #
    # Specification:
    # testMetric = null
    # skip aggregate and slice checks
    # lineage and bytes still apply
    # --------------------------------------------------------

    if len(rows) == 0 or invalid_row:

        test_metric = None
        critical_slice_pass = False

    else:

        # ----------------------------------------------------
        # AGGREGATE ACCURACY
        # ----------------------------------------------------

        correct = 0

        for row in rows:

            if (
                row["label"]
                == row["prediction"]
            ):
                correct += 1

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
        # GROUP TEST ROWS BY SLICE
        # ----------------------------------------------------

        slices = {}

        for row in rows:

            name = row["slice"]

            if name not in slices:
                slices[name] = []

            slices[name].append(row)

        # ----------------------------------------------------
        # REQUIRED SLICES
        # ----------------------------------------------------

        critical_slice_pass = True

        for name in utf8_sorted(
            list(
                data[
                    "requiredSlices"
                ].keys()
            )
        ):

            if name not in slices:

                reasons.append(
                    "MISSING_SLICE:" + name
                )

                critical_slice_pass = False
                continue

            slice_rows = slices[name]

            slice_correct = 0

            for row in slice_rows:

                if (
                    row["label"]
                    == row["prediction"]
                ):
                    slice_correct += 1

            slice_metric = round12(
                slice_correct
                / len(slice_rows)
            )

            floor = float(
                data[
                    "requiredSlices"
                ][name]
            )

            if slice_metric < floor:

                reasons.append(
                    "SLICE_FLOOR:" + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # criticalSlicePass
    #
    # IMPORTANT:
    # This does NOT include aggregate or byte status.
    # --------------------------------------------------------

    if not lineage_valid:
        critical_slice_pass = False

    if invalid_row:
        critical_slice_pass = False

    if len(rows) == 0:
        critical_slice_pass = False

    # --------------------------------------------------------
    # BYTE GATE
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

    # Lineage gate.
    if not lineage_valid:
        decision = "reject"

    # Every test row must be valid and rows non-empty.
    if len(rows) == 0:
        decision = "reject"

    if invalid_row:
        decision = "reject"

    # Aggregate gate.
    if test_metric is None:
        decision = "reject"

    elif (
        test_metric
        < float(data["metricFloor"])
    ):
        decision = "reject"

    # Required slice gate.
    if not critical_slice_pass:
        decision = "reject"

    # Cost gate.
    if (
        data["bytesProcessed"]
        > data["maxBytes"]
    ):
        decision = "reject"

    # --------------------------------------------------------
    # EXACT REQUIRED OUTPUT
    # --------------------------------------------------------

    return {
        "runId": data["runId"],
        "selectedTrialId": data[
            "selectedTrialId"
        ],
        "datasetDigest": data[
            "datasetDigest"
        ],
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": data[
            "bytesProcessed"
        ],
        "reasonCodes": sorted_codes(
            reasons
        )
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

        # ----------------------------------------------------
        # REQUEST MUST BE JSON
        # ----------------------------------------------------

        if not request.is_json:
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        data = request.get_json(
            silent=True
        )

        if not isinstance(
            data,
            dict
        ):
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        phase = data.get("phase")

        # ----------------------------------------------------
        # UNKNOWN / MISSING PHASE
        # ----------------------------------------------------

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

            # Malformed selection still returns the specified
            # selection response rather than a top-level
            # INVALID_INPUT error.
            if not valid_run_id:

                return jsonify(
                    perform_selection(data)
                )

            fingerprint = request_fingerprint(
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

                # ------------------------------------------------
                # SAME ID + DIFFERENT INPUT
                # ------------------------------------------------

                return jsonify({
                    "error":
                        "RUN_ID_CONFLICT"
                }), 409

            # ------------------------------------------------
            # NEW RUN
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
# LOCAL SERVER
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
