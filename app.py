from flask import Flask, request, jsonify
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import re

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

DIGEST_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


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
        INSERT INTO runs
        (run_id, fingerprint, response)
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


def sorted_codes(codes):
    return sorted(
        set(codes),
        key=utf8_key
    )


# ============================================================
# TIMESTAMPS
# ============================================================

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

    except Exception:
        return None


# ============================================================
# JSON DIGESTS
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def dataset_digest(train_ids, eval_ids, features):
    """
    IMPORTANT:
    The assignment requires this exact shape and key order:

    {
      "trainRowIds": ...,
      "evalRowIds": ...,
      "featureNames": ...
    }
    """

    obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features
    }

    return hashlib.sha256(
        compact_json(obj).encode("utf-8")
    ).hexdigest()


def request_fingerprint(data):
    """
    Object key ordering should not make an otherwise identical
    JSON request a RUN_ID conflict.
    """

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
# SELECTION FEATURE VALIDATION
# ============================================================

def valid_feature(feature):
    if not isinstance(feature, dict):
        return False

    # Feature value is DATA.
    # It may be a string, number, boolean, null, object, etc.
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

    if any(
        key not in row
        for key in required
    ):
        return False

    # Row ID.
    if not isinstance(
        row["id"],
        str
    ):
        return False

    if row["id"] == "":
        return False

    # Entity.
    if not isinstance(
        row["entity"],
        str
    ):
        return False

    if row["entity"] == "":
        return False

    # Event time.
    event_time = parse_timestamp(
        row["eventTime"]
    )

    if event_time is None:
        return False

    # Prediction time.
    prediction_time = parse_timestamp(
        row["predictionTime"]
    )

    if prediction_time is None:
        return False

    # The event must not occur after the prediction point.
    #
    # This is an important point-in-time safety rule.
    if event_time > prediction_time:
        return False

    # Version.
    if not safe_integer(
        row["version"]
    ):
        return False

    # Split.
    if row["split"] not in (
        "TRAIN",
        "EVAL"
    ):
        return False

    # Features.
    if not isinstance(
        row["features"],
        dict
    ):
        return False

    for feature_name, feature in row[
        "features"
    ].items():

        if not isinstance(
            feature_name,
            str
        ):
            return False

        if feature_name == "":
            return False

        if not valid_feature(
            feature
        ):
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

        if not valid_selection_row(
            row
        ):
            return False

        # IDs unique in the supplied array.
        if row["id"] in ids:
            return False

        ids.add(row["id"])

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(rows):
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

        # Highest version.
        if row["version"] > old["version"]:
            groups[key] = row
            continue

        # Same version -> UTF-8 smallest ID.
        if (
            row["version"]
            == old["version"]
        ):

            if (
                row["id"].encode("utf-8")
                <
                old["id"].encode("utf-8")
            ):
                groups[key] = row

    return list(groups.values())


# ============================================================
# POINT-IN-TIME FEATURES
# ============================================================

def eligible_features(
    retained_rows,
    forbidden_features
):

    if len(retained_rows) == 0:
        return []

    forbidden = set(
        forbidden_features
    )

    # --------------------------------------------------------
    # Feature must exist in EVERY retained row.
    # --------------------------------------------------------

    shared = set(
        retained_rows[0][
            "features"
        ].keys()
    )

    for row in retained_rows[1:]:

        shared.intersection_update(
            row["features"].keys()
        )

    # --------------------------------------------------------
    # Remove forbidden features.
    # --------------------------------------------------------

    shared = {
        name
        for name in shared
        if name not in forbidden
    }

    result = []

    # --------------------------------------------------------
    # Point-in-time test.
    #
    # For EVERY retained row:
    #
    # availableAt <= predictionTime
    # --------------------------------------------------------

    for feature_name in shared:

        eligible = True

        for row in retained_rows:

            available_at = parse_timestamp(
                row["features"][
                    feature_name
                ]["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            if available_at > prediction_time:
                eligible = False
                break

        if eligible:
            result.append(
                feature_name
            )

    # UTF-8 byte order.
    return utf8_sorted(result)


# ============================================================
# TRIAL VALIDATION
# ============================================================

def valid_trial(trial):
    if not isinstance(
        trial,
        dict
    ):
        return False

    required = (
        "trialId",
        "status",
        "evalMetric"
    )

    if any(
        key not in trial
        for key in required
    ):
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
    if not isinstance(
        trials,
        list
    ):
        return False

    ids = set()

    for trial in trials:

        if not valid_trial(
            trial
        ):
            return False

        if trial["trialId"] in ids:
            return False

        ids.add(
            trial["trialId"]
        )

    return True


# ============================================================
# TRIAL SELECTION
# ============================================================

def select_trial(trials):

    eligible = []

    for trial in trials:

        # Failed trials never participate.
        if trial["status"] != "SUCCEEDED":
            continue

        # Only finite successful trials.
        if not finite_number(
            trial["evalMetric"]
        ):
            continue

        eligible.append(
            trial
        )

    if not eligible:
        return None

    # Highest evalMetric.
    #
    # Exact tie:
    # smallest integer trialId.
    eligible.sort(
        key=lambda trial: (
            -float(
                trial["evalMetric"]
            ),
            trial["trialId"]
        )
    )

    return eligible[0]


# ============================================================
# COMPLETE SELECTION VALIDATION
# ============================================================

def valid_selection(data):

    if not isinstance(
        data,
        dict
    ):
        return False

    if data.get("phase") != "select":
        return False

    run_id = data.get(
        "runId"
    )

    if not isinstance(
        run_id,
        str
    ):
        return False

    if run_id == "":
        return False

    if len(run_id) > 128:
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

    num_trials = data.get(
        "numTrialsLimit"
    )

    if not positive_integer(
        num_trials
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

def perform_selection(data):

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

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
    # DEDUP FIRST
    # --------------------------------------------------------

    retained = deduplicate(
        data["rows"]
    )

    # --------------------------------------------------------
    # SPLIT AFTER DEDUP
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
    # POINT-IN-TIME FEATURES
    # --------------------------------------------------------

    feature_names = eligible_features(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # SELECT TRIAL
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

    # --------------------------------------------------------
    # EXACT RESPONSE SHAPE
    # --------------------------------------------------------

    return {
        "runId": data["runId"],
        "selectedTrialId": selected[
            "trialId"
        ],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": []
    }


# ============================================================
# EVALUATION ROW
# ============================================================

def valid_test_row(row):

    if not isinstance(
        row,
        dict
    ):
        return False

    if row.get("label") not in (
        0,
        1
    ):
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
# EVALUATION INPUT
# ============================================================

def valid_evaluation(data):

    if not isinstance(
        data,
        dict
    ):
        return False

    if data.get("phase") != "evaluate":
        return False

    run_id = data.get(
        "runId"
    )

    if not isinstance(
        run_id,
        str
    ):
        return False

    if run_id == "":
        return False

    if len(run_id) > 128:
        return False

    if not safe_integer(
        data.get("selectedTrialId")
    ):
        return False

    digest = data.get(
        "datasetDigest"
    )

    if (
        not isinstance(
            digest,
            str
        )
        or not DIGEST_RE.fullmatch(
            digest
        )
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

        if not isinstance(
            name,
            str
        ):
            return False

        if name == "":
            return False

        if not finite_number(
            floor
        ):
            return False

        if not (
            0 <= float(floor) <= 1
        ):
            return False

    rows = data.get(
        "rows"
    )

    if not isinstance(
        rows,
        list
    ):
        return False

    if not safe_integer(
        data.get(
            "bytesProcessed"
        )
    ):
        return False

    if not safe_integer(
        data.get(
            "maxBytes"
        )
    ):
        return False

    return True


# ============================================================
# ROUND TO 12 DECIMAL PLACES
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
                data.get(
                    "bytesProcessed",
                    0
                )
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

    stored = load_run(
        data["runId"]
    )

    lineage_valid = False

    if stored is not None:

        saved = stored["response"]

        if (
            saved.get(
                "reasonCodes"
            ) == []
            and
            saved.get(
                "selectedTrialId"
            ) is not None
            and
            saved.get(
                "datasetDigest"
            ) is not None
            and
            saved.get(
                "selectedTrialId"
            )
            == data["selectedTrialId"]
            and
            saved.get(
                "datasetDigest"
            )
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

    invalid_test_row = False

    for row in rows:

        if not valid_test_row(
            row
        ):
            invalid_test_row = True
            break

    if invalid_test_row:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # EMPTY / INVALID TEST DATA
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

        test_metric = round12(
            correct / len(rows)
        )

        if (
            test_metric
            < float(
                data["metricFloor"]
            )
        ):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # GROUP BY SLICE
        # ----------------------------------------------------

        slices = {}

        for row in rows:

            slices.setdefault(
                row["slice"],
                []
            ).append(row)

        critical_slice_pass = True

        # ----------------------------------------------------
        # REQUIRED SLICES
        # ----------------------------------------------------

        for name in utf8_sorted(
            list(
                data[
                    "requiredSlices"
                ].keys()
            )
        ):

            if name not in slices:

                reasons.append(
                    "MISSING_SLICE:"
                    + name
                )

                critical_slice_pass = False
                continue

            slice_rows = slices[name]

            slice_correct = sum(
                1
                for row in slice_rows
                if row["label"]
                == row["prediction"]
            )

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
                    "SLICE_FLOOR:"
                    + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # criticalSlicePass
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

    if invalid_test_row:
        decision = "reject"

    if len(rows) == 0:
        decision = "reject"

    if test_metric is None:
        decision = "reject"

    if (
        test_metric is not None
        and test_metric
        < float(
            data["metricFloor"]
        )
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
    # FINAL RESPONSE
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
        "criticalSlicePass": (
            critical_slice_pass
        ),
        "decision": decision,
        "bytesProcessed": data[
            "bytesProcessed"
        ],
        "reasonCodes": sorted_codes(
            reasons
        )
    }


# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.route(
    "/bqml",
    methods=["POST"]
)
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

        if not isinstance(
            data,
            dict
        ):
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        phase = data.get(
            "phase"
        )

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
        # SELECTION
        # ====================================================

        if phase == "select":

            run_id = data.get(
                "runId"
            )

            # If runId itself is malformed, don't attempt
            # persistence.
            if not (
                isinstance(
                    run_id,
                    str
                )
                and run_id != ""
                and len(run_id) <= 128
            ):

                return jsonify(
                    perform_selection(
                        data
                    )
                )

            fingerprint = request_fingerprint(
                data
            )

            previous = load_run(
                run_id
            )

            # ------------------------------------------------
            # REPLAY
            # ------------------------------------------------

            if previous is not None:

                if (
                    previous[
                        "fingerprint"
                    ]
                    == fingerprint
                ):

                    return jsonify(
                        previous[
                            "response"
                        ]
                    )

                # Same run ID, different input.
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

            return jsonify(
                response
            )

        # ====================================================
        # EVALUATION
        # ====================================================

        return jsonify(
            perform_evaluation(
                data
            )
        )

    except Exception:
        # Never leak a traceback through the API.
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400


# ============================================================
# HEALTH / ROOT
# ============================================================

@app.route("/")
def home():
    return "BQML service running"


# ============================================================
# RENDER ENTRYPOINT
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
