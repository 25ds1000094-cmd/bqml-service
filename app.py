from flask import Flask, request, jsonify
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
import os
import re
import sqlite3

app = Flask(__name__)

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
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            response TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_run(run_id):
    conn = db()

    row = conn.execute(
        "SELECT fingerprint, response FROM runs WHERE run_id = ?",
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
    conn = db()

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
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
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


def sort_codes(values):
    return sorted(set(values), key=utf8_key)


def parse_timestamp(value):
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
    Decimal half-up rounding to 12 decimal places.
    """
    d = Decimal(str(value))
    result = d.quantize(
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

    # The specification does NOT say these must be non-empty.
    if not isinstance(row["id"], str):
        return False

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

    # Validate feature structure.
    #
    # IMPORTANT:
    # feature["value"] is NEVER interpreted.
    # It is arbitrary data.
    for name, feature in row["features"].items():

        if not isinstance(name, str):
            return False

        if not isinstance(feature, dict):
            return False

        if "value" not in feature:
            return False

        if "availableAt" not in feature:
            return False

        if parse_timestamp(feature["availableAt"]) is None:
            return False

    return True


def valid_selection_rows(rows):

    if not isinstance(rows, list):
        return False

    ids = set()

    for row in rows:

        if not valid_selection_row(row):
            return False

        # IDs must be unique in the supplied array.
        if row["id"] in ids:
            return False

        ids.add(row["id"])

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(rows):

    groups = {}

    for row in rows:

        # Parse to UTC first.
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

        # Same version:
        # UTF-8-byte-smallest ID wins.
        elif row["version"] == existing["version"]:

            if utf8_key(row["id"]) < utf8_key(existing["id"]):
                groups[key] = row

    return list(groups.values())


# ============================================================
# POINT-IN-TIME FEATURE ELIGIBILITY
# ============================================================

def eligible_features(rows, forbidden_features):

    if not rows:
        return []

    forbidden = set(forbidden_features)

    # Feature must appear in EVERY retained row.
    shared = set(rows[0]["features"].keys())

    for row in rows[1:]:
        shared.intersection_update(
            row["features"].keys()
        )

    eligible = []

    for feature_name in shared:

        # Forbidden features are removed.
        if feature_name in forbidden:
            continue

        eligible_here = True

        for row in rows:

            feature = row["features"][feature_name]

            available_at = parse_timestamp(
                feature["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # Point-in-time safety:
            #
            # availableAt <= predictionTime
            #
            # Equality is valid.
            if available_at > prediction_time:
                eligible_here = False
                break

        if eligible_here:
            eligible.append(feature_name)

    return utf8_sorted(eligible)


# ============================================================
# TRIALS
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

        # Only finite successful metrics are eligible.
        if not finite_number(trial["evalMetric"]):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest metric.
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
# SELECTION
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

    forbidden = data.get("forbiddenFeatures")

    if not isinstance(forbidden, list):
        return False

    if not all(
        isinstance(x, str)
        for x in forbidden
    ):
        return False

    limit = data.get("numTrialsLimit")

    if not safe_integer(limit) or limit <= 0:
        return False

    rows = data.get("rows")

    if not isinstance(rows, list) or len(rows) == 0:
        return False

    trials = data.get("trials")

    if not valid_selection_rows(rows):
        return False

    if not valid_trials(trials):
        return False

    return True


def make_selection(data):

    # Invalid selection.
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
            "reasonCodes": ["INVALID_INPUT"]
        }

    # Trial limit is a contract failure.
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
    # STEP 1: DEDUPLICATE
    # --------------------------------------------------------

    retained = deduplicate(data["rows"])

    # --------------------------------------------------------
    # STEP 2: SPLIT RETAINED ROW IDS
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
    # STEP 3: SHARED POINT-IN-TIME FEATURES
    # --------------------------------------------------------

    feature_names = eligible_features(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # STEP 4: CHOOSE TRIAL
    # --------------------------------------------------------

    selected = select_trial(data["trials"])

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
    # STEP 5: EXACT DATASET DIGEST OBJECT
    # --------------------------------------------------------

    digest_object = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    dataset_digest = sha256_json(
        digest_object
    )

    return {
        "runId": data["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": []
    }


# ============================================================
# EVALUATION
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

    dataset_digest = data.get(
        "datasetDigest"
    )

    if (
        not isinstance(dataset_digest, str)
        or not DIGEST_RE.fullmatch(dataset_digest)
    ):
        return False

    metric_floor = data.get("metricFloor")

    if not finite_number(metric_floor):
        return False

    if not 0 <= float(metric_floor) <= 1:
        return False

    required_slices = data.get(
        "requiredSlices"
    )

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


def valid_test_row(row):

    if not isinstance(row, dict):
        return False

    if row.get("label") not in (0, 1):
        return False

    if row.get("prediction") not in (0, 1):
        return False

    if not isinstance(row.get("slice"), str):
        return False

    if row["slice"] == "":
        return False

    return True


def make_evaluation(data):

    if not evaluation_valid(data):

        return {
            "runId": (
                data.get("runId", "")
                if isinstance(data, dict)
                else ""
            ),
            "selectedTrialId": (
                data.get("selectedTrialId")
                if isinstance(data, dict)
                and safe_integer(
                    data.get("selectedTrialId")
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
                if isinstance(data, dict)
                and safe_integer(
                    data.get("bytesProcessed")
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

    stored = get_run(data["runId"])

    if stored is None:

        reasons.append(
            "INVALID_LINEAGE"
        )

    else:

        saved = stored["response"]

        if (
            saved["selectedTrialId"] is None
            or saved["selectedTrialId"]
                != data["selectedTrialId"]
            or saved["datasetDigest"]
                != data["datasetDigest"]
        ):
            reasons.append(
                "INVALID_LINEAGE"
            )

    # --------------------------------------------------------
    # TEST ROW VALIDATION
    # --------------------------------------------------------

    rows = data["rows"]

    any_invalid = False

    for row in rows:

        if not valid_test_row(row):
            any_invalid = True
            break

    if any_invalid:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # EMPTY / INVALID TEST DATA
    # --------------------------------------------------------

    if len(rows) == 0 or any_invalid:

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

        test_metric = round_12(
            Decimal(correct)
            / Decimal(len(rows))
        )

        if test_metric < float(
            data["metricFloor"]
        ):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # SLICE ACCURACY
        # ----------------------------------------------------

        grouped = {}

        for row in rows:

            grouped.setdefault(
                row["slice"],
                []
            ).append(row)

        critical_slice_pass = True

        # UTF-8 order isn't needed for calculation,
        # but deterministic processing is useful.
        slice_names = utf8_sorted(
            data["requiredSlices"].keys()
        )

        for slice_name in slice_names:

            floor = float(
                data["requiredSlices"][slice_name]
            )

            if slice_name not in grouped:

                reasons.append(
                    "MISSING_SLICE:"
                    + slice_name
                )

                critical_slice_pass = False
                continue

            group = grouped[slice_name]

            correct_slice = sum(
                1
                for row in group
                if row["label"]
                == row["prediction"]
            )

            slice_accuracy = round_12(
                Decimal(correct_slice)
                / Decimal(len(group))
            )

            if slice_accuracy < floor:

                reasons.append(
                    "SLICE_FLOOR:"
                    + slice_name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # CRITICAL SLICE FLAG
    # --------------------------------------------------------

    if "INVALID_LINEAGE" in reasons:
        critical_slice_pass = False

    if "INVALID_TEST_ROW" in reasons:
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

    return {
        "runId": data["runId"],
        "selectedTrialId": data["selectedTrialId"],
        "datasetDigest": data["datasetDigest"],
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": data["bytesProcessed"],
        "reasonCodes": sort_codes(reasons)
    }


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/bqml")
def bqml():

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

    phase = data.get("phase")

    # Exact required response.
    if phase not in ("select", "evaluate"):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        run_id = data.get("runId")

        # Valid runId is needed for state/replay.
        if (
            isinstance(run_id, str)
            and run_id != ""
            and len(run_id) <= 128
        ):

            fingerprint = sha256_json(data)

            previous = get_run(run_id)

            if previous is not None:

                if (
                    previous["fingerprint"]
                    == fingerprint
                ):
                    return jsonify(
                        previous["response"]
                    )

                return jsonify({
                    "error":
                        "RUN_ID_CONFLICT"
                }), 409

            response = make_selection(data)

            save_run(
                run_id,
                fingerprint,
                response
            )

            return jsonify(response)

        return jsonify(
            make_selection(data)
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    return jsonify(
        make_evaluation(data)
    )


@app.get("/")
def root():
    return "BQML service running"


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", "8080")
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
