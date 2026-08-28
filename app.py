from flask import Flask, request, jsonify
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
import re
import sqlite3

app = Flask(__name__)

# Persistent state.
# Render may restart the Python process, so don't rely on a Python dictionary.
DB_PATH = os.environ.get("DB_PATH", "/tmp/bqml_runs.db")

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


def get_run(run_id):
    conn = get_db()

    row = conn.execute(
        """
        SELECT fingerprint, response
        FROM runs
        WHERE run_id = ?
        """,
        (run_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return {
        "fingerprint": row[0],
        "response": json.loads(row[1])
    }


def save_run(run_id, fingerprint, response):
    conn = get_db()

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
    conn.close()


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_integer(value):
    """
    Safe non-negative integer.
    Used for version and trialId.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def positive_integer(value):
    """
    numTrialsLimit only needs to be a positive integer.
    The assignment does not restrict it to the safe-integer range.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def utf8_key(value):
    return value.encode("utf-8")


def utf8_sorted(values):
    return sorted(values, key=utf8_key)


def sorted_codes(values):
    return sorted(set(values), key=utf8_key)


def parse_timestamp(value):
    """
    Accept exactly:

    YYYY-MM-DDTHH:mm:ssZ
    YYYY-MM-DDTHH:mm:ss.sZ
    YYYY-MM-DDTHH:mm:ss.ssZ
    YYYY-MM-DDTHH:mm:ss.sssZ

    or the same with ±HH:mm.
    """

    if not isinstance(value, str):
        return None

    if not TIMESTAMP_RE.fullmatch(value):
        return None

    try:
        text = value

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def compact_json(value):
    """
    Compact JSON:
    no spaces after separators.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def sha256_json(value):
    return hashlib.sha256(
        compact_json(value).encode("utf-8")
    ).hexdigest()


def round_12(value):
    """
    Round to exactly the required mathematical precision,
    then return JSON-friendly float.
    """

    result = Decimal(str(value)).quantize(
        Decimal("0.000000000001"),
        rounding=ROUND_HALF_UP
    )

    return float(result)


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

def valid_selection_row(row):

    if not isinstance(row, dict):
        return False

    required_fields = [
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    ]

    for field in required_fields:
        if field not in row:
            return False

    # id and entity must be strings.
    #
    # IMPORTANT:
    # The assignment does not require them to be non-empty.
    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    # Timestamps must be valid instants.
    if parse_timestamp(row["eventTime"]) is None:
        return False

    if parse_timestamp(row["predictionTime"]) is None:
        return False

    # Version is a non-negative safe integer.
    if not safe_integer(row["version"]):
        return False

    # Only TRAIN or EVAL.
    if row["split"] not in ("TRAIN", "EVAL"):
        return False

    # Features must be an object.
    if not isinstance(row["features"], dict):
        return False

    # Validate every feature.
    for feature_name, feature_data in row["features"].items():

        if not isinstance(feature_name, str):
            return False

        if not isinstance(feature_data, dict):
            return False

        # Value is DATA.
        # Do not inspect or restrict its type.
        if "value" not in feature_data:
            return False

        if "availableAt" not in feature_data:
            return False

        if parse_timestamp(
            feature_data["availableAt"]
        ) is None:
            return False

    return True


def valid_selection_rows(rows):

    if not isinstance(rows, list):
        return False

    ids = set()

    for row in rows:

        if not valid_selection_row(row):
            return False

        # IDs must be unique within the supplied array.
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

    Keep:
        highest version.

    If version ties:
        UTF-8-byte-smallest ID.
    """

    groups = {}

    for row in rows:

        event_time_utc = parse_timestamp(
            row["eventTime"]
        )

        # Canonical UTC representation.
        utc_key = event_time_utc.isoformat()

        key = (
            row["entity"],
            utc_key
        )

        current = groups.get(key)

        if current is None:
            groups[key] = row
            continue

        # Highest version wins.
        if row["version"] > current["version"]:
            groups[key] = row

        # Same version -> UTF-8-smallest ID.
        elif row["version"] == current["version"]:

            if (
                utf8_key(row["id"])
                < utf8_key(current["id"])
            ):
                groups[key] = row

    return list(groups.values())


# ============================================================
# SHARED / POINT-IN-TIME SAFE FEATURES
# ============================================================

def eligible_features(
    rows,
    forbidden_features
):
    """
    A feature is eligible only when:

    1. It exists in EVERY retained row.
    2. It is not forbidden.
    3. availableAt <= predictionTime in EVERY retained row.
    """

    if not rows:
        return []

    forbidden = set(forbidden_features)

    # Start with all features in the first row.
    shared = set(
        rows[0]["features"].keys()
    )

    # Intersection means it must exist in every row.
    for row in rows[1:]:
        shared.intersection_update(
            row["features"].keys()
        )

    result = []

    for feature_name in shared:

        if feature_name in forbidden:
            continue

        point_in_time_safe = True

        for row in rows:

            feature_data = (
                row["features"][feature_name]
            )

            available_at = parse_timestamp(
                feature_data["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # Leakage check.
            #
            # Equal timestamps are allowed.
            if available_at > prediction_time:
                point_in_time_safe = False
                break

        if point_in_time_safe:
            result.append(feature_name)

    # Required UTF-8-byte ordering.
    return utf8_sorted(result)


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

    # Trial ID must be non-negative safe integer.
    if not safe_integer(
        trial["trialId"]
    ):
        return False

    if trial["status"] not in (
        "SUCCEEDED",
        "FAILED"
    ):
        return False

    # evalMetric is deliberately NOT required to be finite here.
    #
    # The specification says only finite SUCCEEDED trials
    # are eligible. Therefore NaN / Infinity should be
    # ignored during trial selection rather than automatically
    # making the entire request invalid.

    return True


def valid_trials(trials):

    if not isinstance(trials, list):
        return False

    ids = set()

    for trial in trials:

        if not valid_trial(trial):
            return False

        # Trial IDs unique within the array.
        if trial["trialId"] in ids:
            return False

        ids.add(trial["trialId"])

    return True


def select_trial(trials):

    eligible = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        # Only finite successful trials are eligible.
        if not finite_number(
            trial["evalMetric"]
        ):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest evalMetric.
    #
    # Exact tie -> smallest trialId.
    eligible.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    return eligible[0]


# ============================================================
# SELECTION REQUEST VALIDATION
# ============================================================

def selection_valid(data):

    if not isinstance(data, dict):
        return False

    if data.get("phase") != "select":
        return False

    run_id = data.get("runId")

    if (
        not isinstance(run_id, str)
        or run_id == ""
        or len(run_id) > 128
    ):
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

    # Positive integer, not necessarily safe integer.
    limit = data.get(
        "numTrialsLimit"
    )

    if not positive_integer(limit):
        return False

    rows = data.get("rows")

    # Selection rows must be non-empty.
    if not isinstance(rows, list):
        return False

    if len(rows) == 0:
        return False

    trials = data.get("trials")

    if not valid_selection_rows(rows):
        return False

    if not valid_trials(trials):
        return False

    return True


# ============================================================
# CREATE SELECTION RESPONSE
# ============================================================

def make_selection(data):

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not selection_valid(data):

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
    # DEDUPLICATE FIRST
    # --------------------------------------------------------

    retained_rows = deduplicate(
        data["rows"]
    )

    # --------------------------------------------------------
    # TRAIN / EVAL IDS
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
    # SHARED POINT-IN-TIME SAFE FEATURES
    # --------------------------------------------------------

    feature_names = eligible_features(
        retained_rows,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # TRIAL SELECTION
    # --------------------------------------------------------

    selected_trial = select_trial(
        data["trials"]
    )

    if selected_trial is None:

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
    # DATASET DIGEST
    # --------------------------------------------------------

    digest_object = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names
    }

    dataset_digest = sha256_json(
        digest_object
    )

    # EXACTLY the required response fields.
    return {
        "runId": data["runId"],
        "selectedTrialId": selected_trial[
            "trialId"
        ],
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": []
    }


# ============================================================
# EVALUATION REQUEST VALIDATION
# ============================================================

def evaluation_valid(data):

    if not isinstance(data, dict):
        return False

    if data.get("phase") != "evaluate":
        return False

    run_id = data.get("runId")

    if (
        not isinstance(run_id, str)
        or run_id == ""
        or len(run_id) > 128
    ):
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
# TEST ROW VALIDATION
# ============================================================

def valid_test_row(row):

    if not isinstance(row, dict):
        return False

    # Binary integer labels/predictions only.
    if row.get("label") not in (0, 1):
        return False

    if row.get("prediction") not in (
        0,
        1
    ):
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
# EVALUATION
# ============================================================

def make_evaluation(data):

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not evaluation_valid(data):

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
                        data.get(
                            "selectedTrialId"
                        )
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
                        data.get(
                            "bytesProcessed"
                        )
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

    stored = get_run(
        data["runId"]
    )

    if stored is None:

        reasons.append(
            "INVALID_LINEAGE"
        )

    else:

        saved_response = (
            stored["response"]
        )

        # Evaluation must use the exact frozen
        # selected trial and digest.
        if (
            saved_response[
                "selectedTrialId"
            ] is None
            or
            saved_response[
                "selectedTrialId"
            ]
            != data["selectedTrialId"]
            or
            saved_response[
                "datasetDigest"
            ]
            != data["datasetDigest"]
        ):

            reasons.append(
                "INVALID_LINEAGE"
            )

    # --------------------------------------------------------
    # TEST ROWS
    # --------------------------------------------------------

    rows = data["rows"]

    invalid_test_row = False

    for row in rows:

        if not valid_test_row(row):
            invalid_test_row = True
            break

    if invalid_test_row:

        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # EMPTY OR INVALID TEST DATA
    # --------------------------------------------------------

    if (
        len(rows) == 0
        or invalid_test_row
    ):

        test_metric = None
        critical_slice_pass = False

    else:

        # ----------------------------------------------------
        # AGGREGATE ACCURACY
        # ----------------------------------------------------

        correct = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = round_12(
            Decimal(correct)
            / Decimal(len(rows))
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

            slices.setdefault(
                row["slice"],
                []
            ).append(row)

        # Start optimistic.
        critical_slice_pass = True

        # ----------------------------------------------------
        # REQUIRED SLICES
        # ----------------------------------------------------

        for slice_name in utf8_sorted(
            data["requiredSlices"].keys()
        ):

            floor = float(
                data["requiredSlices"][
                    slice_name
                ]
            )

            # Required slice must exist.
            if slice_name not in slices:

                reasons.append(
                    "MISSING_SLICE:"
                    + slice_name
                )

                critical_slice_pass = False
                continue

            slice_rows = slices[
                slice_name
            ]

            slice_correct = sum(
                1
                for row in slice_rows
                if row["label"]
                == row["prediction"]
            )

            slice_accuracy = round_12(
                Decimal(slice_correct)
                / Decimal(len(slice_rows))
            )

            if (
                slice_accuracy
                < floor
            ):

                reasons.append(
                    "SLICE_FLOOR:"
                    + slice_name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # CRITICAL SLICE FLAG
    # --------------------------------------------------------

    if (
        "INVALID_LINEAGE"
        in reasons
    ):
        critical_slice_pass = False

    if (
        "INVALID_TEST_ROW"
        in reasons
    ):
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
    # FINAL DECISION
    # --------------------------------------------------------

    decision = "admit"

    if test_metric is None:
        decision = "reject"

    if reasons:
        decision = "reject"

    if (
        test_metric is not None
        and test_metric
        < float(data["metricFloor"])
    ):
        decision = "reject"

    if (
        data["requiredSlices"]
        and not critical_slice_pass
    ):
        decision = "reject"

    if (
        data["bytesProcessed"]
        > data["maxBytes"]
    ):
        decision = "reject"

    if stored is None:
        decision = "reject"

    # EXACTLY the required output fields.
    return {
        "runId": data["runId"],
        "selectedTrialId": data[
            "selectedTrialId"
        ],
        "datasetDigest": data[
            "datasetDigest"
        ],
        "testMetric": test_metric,
        "criticalSlicePass": (
            critical_slice_pass
        ),
        "decision": decision,
        "bytesProcessed": data[
            "bytesProcessed"
        ],
        "reasonCodes": sort_codes(
            reasons
        )
    }


# ============================================================
# HTTP ENDPOINT
# ============================================================

@app.post("/bqml")
def bqml():

    # --------------------------------------------------------
    # REQUEST MUST BE JSON
    # --------------------------------------------------------

    if not request.is_json:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    try:
        data = request.get_json()

    except Exception:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    if not isinstance(data, dict):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # --------------------------------------------------------
    # PHASE
    # --------------------------------------------------------

    phase = data.get("phase")

    # Unknown OR missing phase:
    # exact required response.
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

        # Stateful replay/conflict handling only makes
        # sense when runId itself has the correct shape.
        if (
            isinstance(run_id, str)
            and run_id != ""
            and len(run_id) <= 128
        ):

            fingerprint = sha256_json(
                data
            )

            previous = get_run(
                run_id
            )

            # Existing exact request:
            # return the previously persisted response unchanged.
            if previous is not None:

                if (
                    previous["fingerprint"]
                    == fingerprint
                ):

                    return jsonify(
                        previous["response"]
                    )

                # Same runId with different input.
                return jsonify({
                    "error":
                        "RUN_ID_CONFLICT"
                }), 409

            response = make_selection(
                data
            )

            # Persist the COMPLETE response.
            save_run(
                run_id,
                fingerprint,
                response
            )

            return jsonify(
                response
            )

        # Invalid runId:
        # make_selection will return INVALID_INPUT.
        return jsonify(
            make_selection(data)
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    return jsonify(
        make_evaluation(data)
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return "BQML service running"


# ============================================================
# LOCAL / RENDER START
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
