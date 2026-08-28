from flask import Flask, request, jsonify
from datetime import datetime, timezone
import hashlib
import json
import math
import re

app = Flask(__name__)

# Stateful storage
runs = {}

SAFE_INT_MAX = 9007199254740991


def is_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_INT_MAX
    )


def finite(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def utf8(x):
    return str(x).encode("utf-8")


def sorted_unique(xs):
    return sorted(set(xs), key=utf8)


def parse_time(value):
    if not isinstance(value, str):
        return None

    pattern = (
        r"^\d{4}-\d{2}-\d{2}T"
        r"\d{2}:\d{2}:\d{2}"
        r"(?:\.\d{1,3})?"
        r"(?:Z|[+-]\d{2}:\d{2})$"
    )

    if not re.fullmatch(pattern, value):
        return None

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        dt = datetime.fromisoformat(value)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def compact_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    )


def digest(obj):
    return hashlib.sha256(
        compact_json(obj).encode("utf-8")
    ).hexdigest()


def codes(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


# ============================================================
# SELECTION
# ============================================================

def valid_selection_request(data):
    if not isinstance(data, dict):
        return False

    if data.get("phase") != "select":
        return False

    run_id = data.get("runId")

    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        return False

    if not isinstance(data.get("forbiddenFeatures"), list):
        return False

    if not all(isinstance(x, str) for x in data["forbiddenFeatures"]):
        return False

    limit = data.get("numTrialsLimit")

    if not is_safe_int(limit) or limit <= 0:
        return False

    if not isinstance(data.get("rows"), list) or not data["rows"]:
        return False

    if not isinstance(data.get("trials"), list):
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
        "features",
    ]

    if any(x not in row for x in required):
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if parse_time(row["eventTime"]) is None:
        return False

    if parse_time(row["predictionTime"]) is None:
        return False

    if not is_safe_int(row["version"]):
        return False

    if row["split"] not in ("TRAIN", "EVAL"):
        return False

    if not isinstance(row["features"], dict):
        return False

    return True


def valid_trial(trial):
    if not isinstance(trial, dict):
        return False

    if not is_safe_int(trial.get("trialId")):
        return False

    if trial.get("status") not in ("SUCCEEDED", "FAILED"):
        return False

    if "evalMetric" not in trial:
        return False

    return True


def deduplicate(rows):
    groups = {}

    for row in rows:
        key = (
            row["entity"],
            parse_time(row["eventTime"])
        )

        if key not in groups:
            groups[key] = row
            continue

        old = groups[key]

        if row["version"] > old["version"]:
            groups[key] = row

        elif (
            row["version"] == old["version"]
            and utf8(row["id"]) < utf8(old["id"])
        ):
            groups[key] = row

    return list(groups.values())


def find_features(rows, forbidden):
    if not rows:
        return []

    forbidden = set(forbidden)

    common = set(rows[0]["features"].keys())

    for row in rows[1:]:
        common &= set(row["features"].keys())

    answer = []

    for name in common:
        if name in forbidden:
            continue

        okay = True

        for row in rows:
            feature = row["features"].get(name)

            if not isinstance(feature, dict):
                okay = False
                break

            available = parse_time(feature.get("availableAt"))
            prediction = parse_time(row["predictionTime"])

            if available is None or prediction is None:
                okay = False
                break

            if available > prediction:
                okay = False
                break

        if okay:
            answer.append(name)

    return sorted(answer, key=utf8)


def choose_trial(trials):
    eligible = []

    for trial in trials:
        if trial["status"] != "SUCCEEDED":
            continue

        if not finite(trial["evalMetric"]):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    eligible.sort(
        key=lambda x: (
            -float(x["evalMetric"]),
            x["trialId"]
        )
    )

    return eligible[0]


def make_selection(data):
    if not valid_selection_request(data):
        return {
            "runId": data.get("runId", ""),
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"]
        }

    reasons = []

    rows = data["rows"]
    trials = data["trials"]

    if len(trials) > data["numTrialsLimit"]:
        reasons.append("TRIAL_LIMIT_EXCEEDED")

    row_ids = set()

    for row in rows:
        if not valid_selection_row(row):
            reasons.append("INVALID_INPUT")
            break

        if row["id"] in row_ids:
            reasons.append("INVALID_INPUT")
            break

        row_ids.add(row["id"])

    trial_ids = set()

    for trial in trials:
        if not valid_trial(trial):
            reasons.append("INVALID_INPUT")
            break

        if trial["trialId"] in trial_ids:
            reasons.append("INVALID_INPUT")
            break

        trial_ids.add(trial["trialId"])

    if reasons:
        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": codes(reasons)
        }

    retained = deduplicate(rows)

    train_ids = sorted(
        [
            r["id"]
            for r in retained
            if r["split"] == "TRAIN"
        ],
        key=utf8
    )

    eval_ids = sorted(
        [
            r["id"]
            for r in retained
            if r["split"] == "EVAL"
        ],
        key=utf8
    )

    feature_names = find_features(
        retained,
        data["forbiddenFeatures"]
    )

    trial = choose_trial(trials)

    if trial is None:
        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"]
        }

    digest_object = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    return {
        "runId": data["runId"],
        "selectedTrialId": trial["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest(digest_object),
        "reasonCodes": []
    }


# ============================================================
# EVALUATION
# ============================================================

def valid_evaluation_request(data):
    if not isinstance(data, dict):
        return False

    if data.get("phase") != "evaluate":
        return False

    if not isinstance(data.get("runId"), str):
        return False

    if not data["runId"] or len(data["runId"]) > 128:
        return False

    if not is_safe_int(data.get("selectedTrialId")):
        return False

    if not isinstance(data.get("datasetDigest"), str):
        return False

    if not re.fullmatch(
        r"[0-9a-f]{64}",
        data["datasetDigest"]
    ):
        return False

    if not finite(data.get("metricFloor")):
        return False

    if not 0 <= float(data["metricFloor"]) <= 1:
        return False

    if not isinstance(data.get("requiredSlices"), dict):
        return False

    for name, floor in data["requiredSlices"].items():
        if not isinstance(name, str) or not name:
            return False

        if not finite(floor):
            return False

        if not 0 <= float(floor) <= 1:
            return False

    if not isinstance(data.get("rows"), list):
        return False

    if not is_safe_int(data.get("bytesProcessed")):
        return False

    if not is_safe_int(data.get("maxBytes")):
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

    if not row["slice"]:
        return False

    return True


def evaluate(data):
    if not valid_evaluation_request(data):
        return {
            "runId": data.get("runId", ""),
            "selectedTrialId": (
                data.get("selectedTrialId")
                if isinstance(data.get("selectedTrialId"), int)
                else None
            ),
            "datasetDigest": data.get("datasetDigest"),
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed": (
                data.get("bytesProcessed")
                if is_safe_int(data.get("bytesProcessed"))
                else 0
            ),
            "reasonCodes": ["INVALID_INPUT"]
        }

    reasons = []

    stored = runs.get(data["runId"])

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

    rows = data["rows"]

    bad_row = any(not valid_test_row(row) for row in rows)

    if bad_row:
        reasons.append("INVALID_TEST_ROW")

    # Empty or invalid rows => no metric/slice calculation.
    if not rows or bad_row:
        test_metric = None
        slice_pass = False

    else:
        correct = sum(
            row["label"] == row["prediction"]
            for row in rows
        )

        test_metric = round(correct / len(rows), 12)

        if test_metric < float(data["metricFloor"]):
            reasons.append("AGGREGATE_FLOOR")

        slice_rows = {}

        for row in rows:
            slice_rows.setdefault(
                row["slice"],
                []
            ).append(row)

        slice_pass = True

        for name, floor in data["requiredSlices"].items():

            if name not in slice_rows:
                reasons.append(
                    f"MISSING_SLICE:{name}"
                )
                slice_pass = False
                continue

            group = slice_rows[name]

            correct_slice = sum(
                row["label"] == row["prediction"]
                for row in group
            )

            accuracy = round(
                correct_slice / len(group),
                12
            )

            if accuracy < float(floor):
                reasons.append(
                    f"SLICE_FLOOR:{name}"
                )
                slice_pass = False

    if data["bytesProcessed"] > data["maxBytes"]:
        reasons.append("BYTE_LIMIT")

    if "INVALID_LINEAGE" in reasons:
        slice_pass = False

    if "INVALID_TEST_ROW" in reasons:
        slice_pass = False

    if not rows:
        slice_pass = False

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

    if data["requiredSlices"] and not slice_pass:
        decision = "reject"

    if data["bytesProcessed"] > data["maxBytes"]:
        decision = "reject"

    return {
        "runId": data["runId"],
        "selectedTrialId": data["selectedTrialId"],
        "datasetDigest": data["datasetDigest"],
        "testMetric": test_metric,
        "criticalSlicePass": slice_pass,
        "decision": decision,
        "bytesProcessed": data["bytesProcessed"],
        "reasonCodes": codes(reasons)
    }


# ============================================================
# HTTP API
# ============================================================

@app.post("/bqml")
def bqml():

    if not request.is_json:
        return jsonify({"error": "INVALID_INPUT"}), 400

    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "INVALID_INPUT"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "INVALID_INPUT"}), 400

    phase = data.get("phase")

    if phase not in ("select", "evaluate"):
        return jsonify({"error": "INVALID_INPUT"}), 400

    if phase == "select":

        run_id = data.get("runId")

        if not (
            isinstance(run_id, str)
            and run_id
            and len(run_id) <= 128
        ):
            return jsonify(make_selection(data))

        fingerprint = digest(data)

        if run_id in runs:

            if runs[run_id]["fingerprint"] == fingerprint:
                return jsonify(
                    runs[run_id]["response"]
                )

            return jsonify({
                "error": "RUN_ID_CONFLICT"
            }), 409

        response = make_selection(data)

        runs[run_id] = {
            "fingerprint": fingerprint,
            "response": response
        }

        return jsonify(response)

    return jsonify(evaluate(data))


@app.get("/")
def home():
    return "BQML service is running"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )
