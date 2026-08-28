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
# DATABASE / STATE
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
# BASIC TYPES
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
        and value > 0
        and value <= MAX_SAFE_INTEGER
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
# TIMESTAMP HANDLING
# ============================================================

def parse_timestamp(value):
    """
    Accept:

    YYYY-MM-DDTHH:mm:ssZ
    YYYY-MM-DDTHH:mm:ss.sZ
    YYYY-MM-DDTHH:mm:ss.ssZ
    YYYY-MM-DDTHH:mm:ss.sssZ

    or an equivalent explicit UTC offset.
    """

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
# HASHING
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def dataset_digest(train_ids, eval_ids, feature_names):
    """
    EXACT required shape and key order.
    """

    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    return hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()


def request_fingerprint(data):
    """
    Used only for runId replay/conflict detection.

    Sorting object keys means equivalent JSON object ordering
    does not create a false conflict.
    """

    payload = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )

    return hashlib.sha256(
        payload.encode("utf-8")
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

    if parse_timestamp(feature["availableAt"]) is None:
        return False

    return True


def valid_selection_row(row):
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

    if any(key not in row for key in required):
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

    event_time = parse_timestamp(
        row["eventTime"]
    )

    if event_time is None:
        return False

    # --------------------------------------------------------
    # PREDICTION TIME
    # --------------------------------------------------------

    prediction_time = parse_timestamp(
        row["predictionTime"]
    )

    if prediction_time is None:
        return False

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    if not safe_integer(row["version"]):
        return False

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    if row["split"] not in ("TRAIN", "EVAL"):
        return False

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

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

    # Selection rows are explicitly non-empty.
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

            if (
                utf8_key(row["id"])
                <
                utf8_key(existing["id"])
            ):
                groups[key] = row

    return list(groups.values())


# ============================================================
# SHARED / POINT-IN-TIME FEATURES
# ============================================================

def calculate_feature_names(
    retained_rows,
    forbidden_features
):
    """
    A feature is eligible iff:

    1. It exists in EVERY retained row.
    2. It is not forbidden.
    3. In EVERY retained row:
           availableAt <= predictionTime

    Feature names are returned in UTF-8 byte order.
    """

    if not retained_rows:
        return []

    forbidden = set(forbidden_features)

    # Start with features from first retained row.
    shared = set(
        retained_rows[0]["features"].keys()
    )

    # Intersection across ALL retained rows.
    for row in retained_rows[1:]:

        shared.intersection_update(
            row["features"].keys()
        )

    # Forbidden features are excluded.
    shared = {
        feature
        for feature in shared
        if feature not in forbidden
    }

    eligible = []

    for feature_name in shared:

        ok = True

        for row in retained_rows:

            feature = row["features"].get(
                feature_name
            )

            # Defensive check.
            if not valid_feature(feature):
                ok = False
                break

            available_at = parse_timestamp(
                feature["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # POINT-IN-TIME LEAKAGE GATE
            if available_at > prediction_time:
                ok = False
                break

        if ok:
            eligible.append(feature_name)

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

    trial_ids = set()

    for trial in trials:

        if not valid_trial(trial):
            return False

        if trial["trialId"] in trial_ids:
            return False

        trial_ids.add(
            trial["trialId"]
        )

    return True


# ============================================================
# MODEL / TRIAL SELECTION
# ============================================================

def choose_trial(trials):
    """
    Only SUCCEEDED + finite trials participate.

    Maximize evalMetric.

    Exact metric tie:
        smallest trialId.
    """

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

    best = eligible[0]

    for trial in eligible[1:]:

        metric = float(
            trial["evalMetric"]
        )

        best_metric = float(
            best["evalMetric"]
        )

        if metric > best_metric:
            best = trial

        elif (
            metric == best_metric
            and trial["trialId"]
            < best["trialId"]
        ):
            best = trial

    return best


# ============================================================
# SELECTION INPUT
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
# SELECTION RESPONSE
# ============================================================

def selection_response(data):

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
    # 1. VALID ROWS
    # 2. DEDUPLICATE
    # --------------------------------------------------------

    retained = deduplicate_rows(
        data["rows"]
    )

    # --------------------------------------------------------
    # 3. SPLIT RETAINED ROWS
    # --------------------------------------------------------

    train_rows = [
        row
        for row in retained
        if row["split"] == "TRAIN"
    ]

    eval_rows = [
        row
        for row in retained
        if row["split"] == "EVAL"
    ]

    # IDs are sorted by UTF-8 bytes.
    train_ids = utf8_sorted([
        row["id"]
        for row in train_rows
    ])

    eval_ids = utf8_sorted([
        row["id"]
        for row in eval_rows
    ])

    # --------------------------------------------------------
    # 4. SHARED POINT-IN-TIME-SAFE FEATURES
    # --------------------------------------------------------

    feature_names = calculate_feature_names(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # 5. SELECT BEST TRIAL
    # --------------------------------------------------------

    selected = choose_trial(
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
# FINAL TEST ROWS
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

    label = row["label"]
    prediction = row["prediction"]

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

    if not isinstance(row["slice"], str):
        return False

    if row["slice"] == "":
        return False

    return True


# ============================================================
# EVALUATION INPUT
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

    selected_trial = data.get(
        "selectedTrialId"
    )

    if not safe_integer(selected_trial):
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
# EVALUATION
# ============================================================

def evaluation_response(data):

    if not validate_evaluation(data):

        run_id = ""
        selected_trial = None
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
                selected_trial = data[
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
            "selectedTrialId": selected_trial,
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

    # ========================================================
    # LINEAGE
    # ========================================================

    stored = load_run(
        data["runId"]
    )

    lineage_valid = False

    if stored is not None:

        saved = stored["response"]

        # Evaluation is permitted ONLY against a successful,
        # persisted selection.
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

    # ========================================================
    # FINAL TEST ROW VALIDATION
    # ========================================================

    rows = data["rows"]

    invalid_test_row = any(
        not valid_test_row(row)
        for row in rows
    )

    if invalid_test_row:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # ========================================================
    # EMPTY / INVALID TEST DATA
    # ========================================================

    if len(rows) == 0 or invalid_test_row:

        test_metric = None

        # Required by specification.
        critical_slice_pass = False

    else:

        # ====================================================
        # AGGREGATE ACCURACY
        # ====================================================

        correct = 0

        for row in rows:

            if row["label"] == row["prediction"]:
                correct += 1

        test_metric = round12(
            correct / len(rows)
        )

        aggregate_pass = (
            test_metric
            >= float(data["metricFloor"])
        )

        if not aggregate_pass:
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ====================================================
        # SLICE ACCURACY
        # ====================================================

        slice_rows = {}

        for row in rows:

            name = row["slice"]

            if name not in slice_rows:
                slice_rows[name] = []

            slice_rows[name].append(row)

        critical_slice_pass = True

        # Required slices are checked in UTF-8 order.
        for name in utf8_sorted(
            list(data["requiredSlices"].keys())
        ):

            if name not in slice_rows:

                reasons.append(
                    "MISSING_SLICE:" + name
                )

                critical_slice_pass = False
                continue

            current_rows = slice_rows[name]

            slice_correct = sum(
                1
                for row in current_rows
                if row["label"]
                == row["prediction"]
            )

            slice_metric = round12(
                slice_correct
                / len(current_rows)
            )

            floor = float(
                data["requiredSlices"][name]
            )

            if slice_metric < floor:

                reasons.append(
                    "SLICE_FLOOR:" + name
                )

                critical_slice_pass = False

    # ========================================================
    # CRITICAL SLICE PASS IS ONLY A SLICE/VALIDITY FLAG
    # ========================================================

    if not lineage_valid:
        critical_slice_pass = False

    if invalid_test_row:
        critical_slice_pass = False

    if len(rows) == 0:
        critical_slice_pass = False

    # ========================================================
    # BYTE GATE
    # ========================================================

    bytes_pass = (
        data["bytesProcessed"]
        <= data["maxBytes"]
    )

    if not bytes_pass:
        reasons.append(
            "BYTE_LIMIT"
        )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    rows_valid = (
        len(rows) > 0
        and not invalid_test_row
    )

    aggregate_pass = (
        test_metric is not None
        and test_metric
        >= float(data["metricFloor"])
    )

    decision = "admit"

    if not lineage_valid:
        decision = "reject"

    if not rows_valid:
        decision = "reject"

    if not aggregate_pass:
        decision = "reject"

    if not critical_slice_pass:
        decision = "reject"

    if not bytes_pass:
        decision = "reject"

    # ========================================================
    # EXACT REQUIRED OUTPUT
    # ========================================================

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

@app.route("/bqml", methods=["POST"])
def bqml():

    try:

        # ----------------------------------------------------
        # JSON ONLY
        # ----------------------------------------------------

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

            # A malformed runId cannot participate in
            # persistence/replay.
            valid_run_id = (
                isinstance(run_id, str)
                and run_id != ""
                and len(run_id) <= 128
            )

            if not valid_run_id:
                return jsonify(
                    selection_response(data)
                )

            fingerprint = request_fingerprint(
                data
            )

            previous = load_run(
                run_id
            )

            # ------------------------------------------------
            # SAME INPUT = EXACT REPLAY
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
                # SAME RUN ID + DIFFERENT INPUT
                # ------------------------------------------------

                return jsonify({
                    "error":
                        "RUN_ID_CONFLICT"
                }), 409

            # ------------------------------------------------
            # NEW RUN
            # ------------------------------------------------

            response = selection_response(
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
            evaluation_response(data)
        )

    except Exception:
        # Keep the endpoint JSON-only and do not expose
        # implementation details.
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
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
