from flask import Flask, request, jsonify
from datetime import datetime, timezone
import hashlib
import json
import math
import re

app = Flask(__name__)

# runId -> frozen successful/unsuccessful selection response
RUNS = {}

MAX_SAFE_INT = 9007199254740991

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= MAX_SAFE_INT
    )


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def utf8_key(x):
    return x.encode("utf-8")


def sorted_utf8(values):
    return sorted(values, key=utf8_key)


def unique_sorted_utf8(values):
    return sorted(set(values), key=utf8_key)


def sorted_codes(values):
    return sorted(set(values), key=utf8_key)


def compact_json(obj):
    # Exact compact JSON: no spaces.
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_compact_json(obj):
    payload = compact_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_instant(value):
    """
    Strictly accept:

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
# SELECTION ROW VALIDATION
# ============================================================

def validate_selection_row(row):
    if not isinstance(row, dict):
        return False

    required = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    }

    # Require the expected fields.
    if not required.issubset(row.keys()):
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    # Empty IDs/entities are not useful row identifiers.
    if row["id"] == "":
        return False

    if row["entity"] == "":
        return False

    # Both timestamps must be valid instants.
    if parse_instant(row["eventTime"]) is None:
        return False

    if parse_instant(row["predictionTime"]) is None:
        return False

    if not safe_int(row["version"]):
        return False

    if row["split"] not in ("TRAIN", "EVAL"):
        return False

    if not isinstance(row["features"], dict):
        return False

    # Validate every feature object now.
    #
    # This is important: don't silently ignore malformed feature data.
    for feature_name, feature_data in row["features"].items():

        if not isinstance(feature_name, str):
            return False

        if not isinstance(feature_data, dict):
            return False

        if "value" not in feature_data:
            return False

        if "availableAt" not in feature_data:
            return False

        if parse_instant(feature_data["availableAt"]) is None:
            return False

    return True


def validate_all_selection_rows(rows):
    if not isinstance(rows, list):
        return False

    # Row IDs must be unique within the supplied array.
    ids = set()

    for row in rows:
        if not validate_selection_row(row):
            return False

        row_id = row["id"]

        if row_id in ids:
            return False

        ids.add(row_id)

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_rows(rows):
    """
    Deduplicate by:

        [entity, UTC(eventTime)]

    Keep:
        highest integer version

    If versions tie:
        UTF-8-byte-smallest ID
    """

    groups = {}

    for row in rows:
        event_utc = parse_instant(row["eventTime"])

        key = (
            row["entity"],
            event_utc,
        )

        current = groups.get(key)

        if current is None:
            groups[key] = row
            continue

        if row["version"] > current["version"]:
            groups[key] = row

        elif row["version"] == current["version"]:
            if utf8_key(row["id"]) < utf8_key(current["id"]):
                groups[key] = row

    return list(groups.values())


# ============================================================
# POINT-IN-TIME FEATURE CHECK
# ============================================================

def find_shared_eligible_features(rows, forbidden_features):
    """
    A feature is eligible only if:

    1. It appears in EVERY retained row.
    2. It is not forbidden.
    3. availableAt <= predictionTime in EVERY retained row.
    """

    if not rows:
        return []

    forbidden = set(forbidden_features)

    # Start with all features from the first retained row.
    shared = set(rows[0]["features"].keys())

    # Intersection means the feature must appear everywhere.
    for row in rows[1:]:
        shared.intersection_update(row["features"].keys())

    eligible = []

    for name in shared:

        if name in forbidden:
            continue

        feature_is_safe = True

        for row in rows:
            feature_data = row["features"][name]

            available_at = parse_instant(
                feature_data["availableAt"]
            )

            prediction_time = parse_instant(
                row["predictionTime"]
            )

            # Point-in-time rule.
            #
            # Equality is allowed.
            if available_at > prediction_time:
                feature_is_safe = False
                break

        if feature_is_safe:
            eligible.append(name)

    return sorted_utf8(eligible)


# ============================================================
# TRIAL VALIDATION / SELECTION
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

    if not safe_int(trial["trialId"]):
        return False

    if trial["status"] not in ("SUCCEEDED", "FAILED"):
        return False

    # evalMetric itself can be non-finite.
    # Those trials are simply ineligible later.
    #
    # Therefore don't reject the whole request here.
    return True


def validate_all_trials(trials):
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


def choose_trial(trials):
    eligible = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        if not finite_number(trial["evalMetric"]):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest metric first.
    #
    # If metrics are exactly tied:
    # smallest integer trialId wins.
    eligible.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"],
        )
    )

    return eligible[0]


# ============================================================
# SELECTION REQUEST VALIDATION
# ============================================================

def validate_selection_request(data):
    if not isinstance(data, dict):
        return False

    if data.get("phase") != "select":
        return False

    run_id = data.get("runId")

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return False

    forbidden = data.get("forbiddenFeatures")

    if not isinstance(forbidden, list):
        return False

    if not all(isinstance(x, str) for x in forbidden):
        return False

    limit = data.get("numTrialsLimit")

    if not safe_int(limit) or limit <= 0:
        return False

    rows = data.get("rows")

    if not isinstance(rows, list) or not rows:
        return False

    trials = data.get("trials")

    if not isinstance(trials, list):
        return False

    if not validate_all_selection_rows(rows):
        return False

    if not validate_all_trials(trials):
        return False

    return True


# ============================================================
# BUILD SELECTION RESPONSE
# ============================================================

def make_selection(data):

    if not validate_selection_request(data):
        return {
            "runId": data.get("runId", "")
            if isinstance(data, dict)
            else "",
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    if len(data["trials"]) > data["numTrialsLimit"]:
        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["TRIAL_LIMIT_EXCEEDED"],
        }

    retained = deduplicate_rows(data["rows"])

    train_ids = [
        row["id"]
        for row in retained
        if row["split"] == "TRAIN"
    ]

    eval_ids = [
        row["id"]
        for row in retained
        if row["split"] == "EVAL"
    ]

    train_ids = sorted_utf8(train_ids)
    eval_ids = sorted_utf8(eval_ids)

    feature_names = find_shared_eligible_features(
        retained,
        data["forbiddenFeatures"],
    )

    selected = choose_trial(data["trials"])

    if selected is None:
        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"],
        }

    digest_object = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    dataset_digest = sha256_compact_json(digest_object)

    return {
        "runId": data["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": [],
    }


# ============================================================
# EVALUATION REQUEST VALIDATION
# ============================================================

def validate_evaluation_request(data):

    if not isinstance(data, dict):
        return False

    if data.get("phase") != "evaluate":
        return False

    run_id = data.get("runId")

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return False

    if not safe_int(data.get("selectedTrialId")):
        return False

    digest_value = data.get("datasetDigest")

    if (
        not isinstance(digest_value, str)
        or not DIGEST_RE.fullmatch(digest_value)
    ):
        return False

    metric_floor = data.get("metricFloor")

    if not finite_number(metric_floor):
        return False

    if not 0 <= float(metric_floor) <= 1:
        return False

    required_slices = data.get("requiredSlices")

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():

        if not isinstance(name, str) or not name:
            return False

        if not finite_number(floor):
            return False

        if not 0 <= float(floor) <= 1:
            return False

    rows = data.get("rows")

    if not isinstance(rows, list):
        return False

    if not safe_int(data.get("bytesProcessed")):
        return False

    if not safe_int(data.get("maxBytes")):
        return False

    return True


def validate_test_row(row):
    if not isinstance(row, dict):
        return False

    if "label" not in row:
        return False

    if "prediction" not in row:
        return False

    if "slice" not in row:
        return False

    if row["label"] not in (0, 1):
        return False

    if row["prediction"] not in (0, 1):
        return False

    if not isinstance(row["slice"], str):
        return False

    if not row["slice"]:
        return False

    return True


# ============================================================
# EVALUATION
# ============================================================

def make_evaluation(data):

    if not validate_evaluation_request(data):
        return {
            "runId": data.get("runId", "")
            if isinstance(data, dict)
            else "",
            "selectedTrialId": (
                data.get("selectedTrialId")
                if isinstance(data, dict)
                and safe_int(data.get("selectedTrialId"))
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
                and safe_int(data.get("bytesProcessed"))
                else 0
            ),
            "reasonCodes": ["INVALID_INPUT"],
        }

    reasons = []

    # --------------------------------------------------------
    # FROZEN LINEAGE
    # --------------------------------------------------------

    stored = RUNS.get(data["runId"])

    if stored is None:
        reasons.append("INVALID_LINEAGE")

    else:
        saved = stored["response"]

        if (
            saved["selectedTrialId"] is None
            or saved["selectedTrialId"] != data["selectedTrialId"]
            or saved["datasetDigest"] != data["datasetDigest"]
        ):
            reasons.append("INVALID_LINEAGE")

    # --------------------------------------------------------
    # FINAL TEST ROWS
    # --------------------------------------------------------

    rows = data["rows"]

    invalid_row = False

    for row in rows:
        if not validate_test_row(row):
            invalid_row = True
            break

    if invalid_row:
        reasons.append("INVALID_TEST_ROW")

    # If empty or invalid:
    #
    # - testMetric = null
    # - skip aggregate calculation
    # - skip slice calculation
    #
    # Lineage and byte gates still happen.
    if not rows or invalid_row:

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

        test_metric = round(
            correct / len(rows),
            12
        )

        if test_metric < float(data["metricFloor"]):
            reasons.append("AGGREGATE_FLOOR")

        # ----------------------------------------------------
        # REQUIRED SLICES
        # ----------------------------------------------------

        groups = {}

        for row in rows:
            groups.setdefault(
                row["slice"],
                []
            ).append(row)

        critical_slice_pass = True

        for slice_name in sorted_utf8(
            data["requiredSlices"].keys()
        ):

            floor = float(
                data["requiredSlices"][slice_name]
            )

            if slice_name not in groups:
                reasons.append(
                    f"MISSING_SLICE:{slice_name}"
                )
                critical_slice_pass = False
                continue

            group = groups[slice_name]

            correct_slice = sum(
                1
                for row in group
                if row["label"] == row["prediction"]
            )

            slice_accuracy = round(
                correct_slice / len(group),
                12
            )

            if slice_accuracy < floor:
                reasons.append(
                    f"SLICE_FLOOR:{slice_name}"
                )
                critical_slice_pass = False

    # --------------------------------------------------------
    # BYTE LIMIT
    # --------------------------------------------------------

    if data["bytesProcessed"] > data["maxBytes"]:
        reasons.append("BYTE_LIMIT")

    # --------------------------------------------------------
    # SLICE PASS FLAG
    # --------------------------------------------------------

    if "INVALID_LINEAGE" in reasons:
        critical_slice_pass = False

    if "INVALID_TEST_ROW" in reasons:
        critical_slice_pass = False

    if not rows:
        critical_slice_pass = False

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = "admit"

    if reasons:
        decision = "reject"

    if test_metric is None:
        decision = "reject"

    if (
        test_metric is not None
        and test_metric < float(data["metricFloor"])
    ):
        decision = "reject"

    if data["requiredSlices"] and not critical_slice_pass:
        decision = "reject"

    if data["bytesProcessed"] > data["maxBytes"]:
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
        "reasonCodes": sorted_codes(reasons),
    }


# ============================================================
# HTTP ENDPOINT
# ============================================================

@app.post("/bqml")
def bqml():

    # Must receive JSON.
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

    # Missing/unknown phase has an EXACT response.
    if phase not in ("select", "evaluate"):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    if phase == "select":

        run_id = data.get("runId")

        # For a valid runId, enforce state/replay/conflict.
        if (
            isinstance(run_id, str)
            and run_id
            and len(run_id) <= 128
        ):

            fingerprint = sha256_compact_json(data)

            if run_id in RUNS:

                saved = RUNS[run_id]

                if saved["fingerprint"] == fingerprint:
                    return jsonify(
                        saved["response"]
                    )

                return jsonify({
                    "error": "RUN_ID_CONFLICT"
                }), 409

            response = make_selection(data)

            # Persist the COMPLETE response.
            RUNS[run_id] = {
                "fingerprint": fingerprint,
                "response": response,
            }

            return jsonify(response)

        return jsonify(make_selection(data))

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    return jsonify(make_evaluation(data))


@app.get("/")
def root():
    return "BQML service running"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )
