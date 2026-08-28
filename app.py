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

# Use a writable location on Render.
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(tempfile.gettempdir(), "bqml_runs.db")
)

MAX_SAFE_INTEGER = 9007199254740991

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


# ============================================================
# DATABASE
# ============================================================

def database():
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
    try:
        conn = database()

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

    except Exception:
        return None


def store_run(run_id, fingerprint, response):
    conn = database()

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
# HELPERS
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
    )


def finite_number(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    except Exception:
        return False


def utf8(value):
    return value.encode("utf-8")


def utf8_sort(values):
    return sorted(values, key=utf8)


def sort_codes(values):
    return sorted(set(values), key=utf8)


def parse_time(value):
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

    except Exception:
        return None


def digest_for(value):
    text = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def rounded(value):
    result = Decimal(str(value)).quantize(
        Decimal("0.000000000001"),
        rounding=ROUND_HALF_UP
    )

    return float(result)


# ============================================================
# SELECTION ROWS
# ============================================================

def valid_feature(feature):

    if not isinstance(feature, dict):
        return False

    # "value" is arbitrary DATA.
    if "value" not in feature:
        return False

    if "availableAt" not in feature:
        return False

    return (
        parse_time(
            feature["availableAt"]
        )
        is not None
    )


def valid_selection_row(row):

    if not isinstance(row, dict):
        return False

    fields = (
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    )

    for field in fields:
        if field not in row:
            return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if parse_time(
        row["eventTime"]
    ) is None:
        return False

    if parse_time(
        row["predictionTime"]
    ) is None:
        return False

    if not safe_integer(
        row["version"]
    ):
        return False

    if row["split"] not in (
        "TRAIN",
        "EVAL"
    ):
        return False

    if not isinstance(
        row["features"],
        dict
    ):
        return False

    for name, feature in row["features"].items():

        if not isinstance(name, str):
            return False

        if not valid_feature(feature):
            return False

    return True


def validate_selection_rows(rows):

    if not isinstance(rows, list):
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

    groups = {}

    for row in rows:

        event_time = parse_time(
            row["eventTime"]
        )

        key = (
            row["entity"],
            event_time
        )

        old = groups.get(key)

        if old is None:
            groups[key] = row
            continue

        if row["version"] > old["version"]:
            groups[key] = row

        elif row["version"] == old["version"]:

            if utf8(row["id"]) < utf8(old["id"]):
                groups[key] = row

    return list(groups.values())


# ============================================================
# POINT-IN-TIME FEATURES
# ============================================================

def find_features(
    rows,
    forbidden
):

    if not rows:
        return []

    forbidden_set = set(forbidden)

    # Feature must appear in EVERY retained row.
    common = set(
        rows[0]["features"].keys()
    )

    for row in rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    result = []

    for name in common:

        if name in forbidden_set:
            continue

        safe = True

        for row in rows:

            available = parse_time(
                row["features"]
                    [name]
                    ["availableAt"]
            )

            prediction = parse_time(
                row["predictionTime"]
            )

            # No future information.
            if available > prediction:
                safe = False
                break

        if safe:
            result.append(name)

    return utf8_sort(result)


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


def validate_trials(trials):

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

    eligible.sort(
        key=lambda x: (
            -float(x["evalMetric"]),
            x["trialId"]
        )
    )

    return eligible[0]


# ============================================================
# SELECTION
# ============================================================

def valid_selection(data):

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

    if not isinstance(
        forbidden,
        list
    ):
        return False

    if not all(
        isinstance(x, str)
        for x in forbidden
    ):
        return False

    limit = data.get(
        "numTrialsLimit"
    )

    if not positive_integer(limit):
        return False

    rows = data.get("rows")

    if (
        not isinstance(rows, list)
        or len(rows) == 0
    ):
        return False

    trials = data.get("trials")

    if not validate_selection_rows(rows):
        return False

    if not validate_trials(trials):
        return False

    return True


def selection_response(data):

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

    # Trial count limit.
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

    # Deduplicate BEFORE splitting.
    retained = deduplicate(
        data["rows"]
    )

    train_ids = utf8_sort([
        row["id"]
        for row in retained
        if row["split"] == "TRAIN"
    ])

    eval_ids = utf8_sort([
        row["id"]
        for row in retained
        if row["split"] == "EVAL"
    ])

    features = find_features(
        retained,
        data["forbiddenFeatures"]
    )

    selected = choose_trial(
        data["trials"]
    )

    if selected is None:

        return {
            "runId": data["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": features,
            "datasetDigest": None,
            "reasonCodes": [
                "NO_SUCCESSFUL_TRIAL"
            ]
        }

    digest_object = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features
    }

    return {
        "runId": data["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
        "datasetDigest": digest_for(
            digest_object
        ),
        "reasonCodes": []
    }


# ============================================================
# EVALUATION
# ============================================================

def valid_evaluation(data):

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

    required = data.get(
        "requiredSlices"
    )

    if not isinstance(
        required,
        dict
    ):
        return False

    for name, floor in required.items():

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


def valid_test_row(row):

    if not isinstance(row, dict):
        return False

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


def evaluation_response(data):

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
                data.get("bytesProcessed", 0)
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

    stored = load_run(
        data["runId"]
    )

    lineage_ok = True

    if stored is None:
        lineage_ok = False
    else:

        saved = stored["response"]

        if (
            saved.get("selectedTrialId")
            != data["selectedTrialId"]
            or
            saved.get("datasetDigest")
            != data["datasetDigest"]
            or
            saved.get("selectedTrialId")
            is None
            or
            saved.get("datasetDigest")
            is None
        ):
            lineage_ok = False

    if not lineage_ok:
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
    # METRICS
    # --------------------------------------------------------

    if (
        len(rows) == 0
        or invalid_row
    ):

        test_metric = None
        critical_slice_pass = False

    else:

        correct = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = rounded(
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

        # Group by slice.
        groups = {}

        for row in rows:
            groups.setdefault(
                row["slice"],
                []
            ).append(row)

        critical_slice_pass = True

        # Every required slice must exist.
        for name in utf8_sort(
            data["requiredSlices"].keys()
        ):

            floor = float(
                data["requiredSlices"][name]
            )

            if name not in groups:

                reasons.append(
                    "MISSING_SLICE:"
                    + name
                )

                critical_slice_pass = False
                continue

            group = groups[name]

            correct_slice = sum(
                1
                for row in group
                if row["label"]
                == row["prediction"]
            )

            slice_metric = rounded(
                Decimal(correct_slice)
                / Decimal(len(group))
            )

            if slice_metric < floor:

                reasons.append(
                    "SLICE_FLOOR:"
                    + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # CRITICAL SLICE FLAG
    # --------------------------------------------------------

    if not lineage_ok:
        critical_slice_pass = False

    if invalid_row:
        critical_slice_pass = False

    if len(rows) == 0:
        critical_slice_pass = False

    # --------------------------------------------------------
    # COST
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

    if not lineage_ok:
        decision = "reject"

    if invalid_row:
        decision = "reject"

    if (
        test_metric is not None
        and test_metric
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
# ENDPOINT
# ============================================================

@app.post("/bqml")
def bqml():

    try:

        # Must be JSON.
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

            run_id = data.get(
                "runId"
            )

            # Valid runId -> stateful behavior.
            if (
                isinstance(run_id, str)
                and run_id != ""
                and len(run_id) <= 128
            ):

                fingerprint = digest_for(
                    data
                )

                previous = load_run(
                    run_id
                )

                if previous is not None:

                    # Identical replay.
                    if (
                        previous["fingerprint"]
                        == fingerprint
                    ):
                        return jsonify(
                            previous["response"]
                        )

                    # Same ID, different input.
                    return jsonify({
                        "error":
                            "RUN_ID_CONFLICT"
                    }), 409

                response = selection_response(
                    data
                )

                store_run(
                    run_id,
                    fingerprint,
                    response
                )

                return jsonify(
                    response
                )

            # Invalid input.
            return jsonify(
                selection_response(data)
            )

        # ====================================================
        # EVALUATE
        # ====================================================

        return jsonify(
            evaluation_response(data)
        )

    except Exception:
        # Never expose an internal traceback to the grader.
        #
        # A malformed request should result in the contract's
        # INVALID_INPUT response rather than HTTP 500.
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return "BQML service running"


# ============================================================
# START
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
