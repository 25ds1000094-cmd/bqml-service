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

# ============================================================
# CONFIG
# ============================================================

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(
        tempfile.gettempdir(),
        "bqml_runs.sqlite3"
    )
)

SAFE_MAX = 9007199254740991

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


# ============================================================
# DATABASE
# ============================================================

def db():
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
    conn = db()

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


def put_run(run_id, fingerprint, response):
    conn = db()

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
# BASIC TYPES
# ============================================================

def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value <= SAFE_MAX
    )


def is_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def is_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def utf8_sort(values):
    return sorted(
        values,
        key=lambda x: x.encode("utf-8")
    )


def sort_reason_codes(codes):
    return sorted(
        set(codes),
        key=lambda x: x.encode("utf-8")
    )


# ============================================================
# TIMESTAMP
# ============================================================

def parse_instant(value):
    """
    Valid forms:

    YYYY-MM-DDTHH:mm:ssZ
    YYYY-MM-DDTHH:mm:ss.sZ
    YYYY-MM-DDTHH:mm:ss.ssZ
    YYYY-MM-DDTHH:mm:ss.sssZ

    or an explicit +/-HH:mm offset.

    Returns a UTC-aware datetime.
    """

    if not isinstance(value, str):
        return None

    if not TIMESTAMP_PATTERN.fullmatch(value):
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


# ============================================================
# JSON / DIGEST
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def sha256_compact_json(value):
    raw = compact_json(value).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

def validate_feature(feature):

    if not isinstance(feature, dict):
        return False

    # The assignment explicitly treats feature values as DATA.
    # Therefore we only require that "value" exists.
    if "value" not in feature:
        return False

    if "availableAt" not in feature:
        return False

    if parse_instant(
        feature["availableAt"]
    ) is None:
        return False

    return True


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

    # ID and entity are strings.
    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    # Both timestamps must be valid instants.
    if parse_instant(
        row["eventTime"]
    ) is None:
        return False

    if parse_instant(
        row["predictionTime"]
    ) is None:
        return False

    # Version is non-negative safe integer.
    if not is_safe_integer(
        row["version"]
    ):
        return False

    # Only these split values.
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

    for feature_name, feature in row["features"].items():

        if not isinstance(
            feature_name,
            str
        ):
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

    seen_ids = set()

    for row in rows:

        if not validate_selection_row(
            row
        ):
            return False

        # IDs unique within supplied array.
        if row["id"] in seen_ids:
            return False

        seen_ids.add(row["id"])

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_rows(rows):
    """
    Deduplicate using:

        [entity, UTC(eventTime)]

    Winner:

        1. highest version
        2. if same version, UTF-8-smallest ID
    """

    groups = {}

    for row in rows:

        utc_event = parse_instant(
            row["eventTime"]
        )

        # datetime objects are timezone-aware UTC instants.
        key = (
            row["entity"],
            utc_event
        )

        current = groups.get(key)

        if current is None:
            groups[key] = row
            continue

        # Highest version wins.
        if (
            row["version"]
            > current["version"]
        ):
            groups[key] = row
            continue

        # Same version -> smallest UTF-8 ID.
        if (
            row["version"]
            == current["version"]
        ):
            if (
                row["id"].encode("utf-8")
                <
                current["id"].encode("utf-8")
            ):
                groups[key] = row

    return list(groups.values())


# ============================================================
# POINT-IN-TIME FEATURE SELECTION
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

    # --------------------------------------------------------
    # STEP 1:
    # Feature must occur in every retained row.
    # --------------------------------------------------------

    shared = set(
        retained_rows[0][
            "features"
        ].keys()
    )

    for row in retained_rows[1:]:

        shared = shared.intersection(
            row["features"].keys()
        )

    # --------------------------------------------------------
    # STEP 2:
    # Remove forbidden features.
    # --------------------------------------------------------

    shared = {
        name
        for name in shared
        if name not in forbidden
    }

    # --------------------------------------------------------
    # STEP 3:
    # Point-in-time check.
    #
    # EVERY occurrence must have:
    #
    # availableAt <= predictionTime
    # --------------------------------------------------------

    eligible = []

    for feature_name in shared:

        feature_is_safe = True

        for row in retained_rows:

            available_at = parse_instant(
                row["features"][
                    feature_name
                ]["availableAt"]
            )

            prediction_time = parse_instant(
                row["predictionTime"]
            )

            if available_at > prediction_time:
                feature_is_safe = False
                break

        if feature_is_safe:
            eligible.append(
                feature_name
            )

    # --------------------------------------------------------
    # STEP 4:
    # UTF-8 byte ordering.
    # --------------------------------------------------------

    return utf8_sort(
        eligible
    )


# ============================================================
# TRIAL VALIDATION
# ============================================================

def validate_trial(trial):

    if not isinstance(
        trial,
        dict
    ):
        return False

    if "trialId" not in trial:
        return False

    if "status" not in trial:
        return False

    if "evalMetric" not in trial:
        return False

    if not is_safe_integer(
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

    if not isinstance(
        trials,
        list
    ):
        return False

    seen = set()

    for trial in trials:

        if not validate_trial(
            trial
        ):
            return False

        trial_id = trial["trialId"]

        if trial_id in seen:
            return False

        seen.add(trial_id)

    return True


# ============================================================
# SELECT TRIAL
# ============================================================

def choose_trial(trials):

    candidates = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        # Only FINITE successful trials are eligible.
        if not is_finite(
            trial["evalMetric"]
        ):
            continue

        candidates.append(
            trial
        )

    if not candidates:
        return None

    # Highest metric.
    # Exact tie -> smallest trialId.
    candidates.sort(
        key=lambda t: (
            -float(t["evalMetric"]),
            t["trialId"]
        )
    )

    return candidates[0]


# ============================================================
# SELECTION INPUT
# ============================================================

def validate_selection(data):

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

    if not is_positive_integer(
        limit
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
# BUILD SELECTION RESPONSE
# ============================================================

def build_selection(data):

    if not validate_selection(
        data
    ):

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

    retained = deduplicate_rows(
        data["rows"]
    )

    # --------------------------------------------------------
    # SPLIT AFTER DEDUPLICATION
    # --------------------------------------------------------

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

    train_ids = utf8_sort(
        train_ids
    )

    eval_ids = utf8_sort(
        eval_ids
    )

    # --------------------------------------------------------
    # POINT-IN-TIME SAFE FEATURES
    # --------------------------------------------------------

    features = eligible_feature_names(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # TRIAL
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
            "featureNames": features,
            "datasetDigest": None,
            "reasonCodes": [
                "NO_SUCCESSFUL_TRIAL"
            ]
        }

    # --------------------------------------------------------
    # EXACT DIGEST OBJECT
    # --------------------------------------------------------

    digest_data = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features
    }

    dataset_digest = sha256_compact_json(
        digest_data
    )

    # --------------------------------------------------------
    # EXACT SELECTION RESPONSE
    # --------------------------------------------------------

    return {
        "runId": data["runId"],
        "selectedTrialId": selected[
            "trialId"
        ],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
        "datasetDigest": dataset_digest,
        "reasonCodes": []
    }


# ============================================================
# TEST ROW
# ============================================================

def validate_test_row(row):

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

def validate_evaluation(data):

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

    if (
        not isinstance(run_id, str)
        or run_id == ""
        or len(run_id) > 128
    ):
        return False

    if not is_safe_integer(
        data.get("selectedTrialId")
    ):
        return False

    digest = data.get(
        "datasetDigest"
    )

    if (
        not isinstance(digest, str)
        or not DIGEST_PATTERN.fullmatch(
            digest
        )
    ):
        return False

    metric_floor = data.get(
        "metricFloor"
    )

    if not is_finite(
        metric_floor
    ):
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

        if not isinstance(
            name,
            str
        ):
            return False

        if name == "":
            return False

        if not is_finite(
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

    if not is_safe_integer(
        data.get("bytesProcessed")
    ):
        return False

    if not is_safe_integer(
        data.get("maxBytes")
    ):
        return False

    return True


# ============================================================
# EVALUATION
# ============================================================

def build_evaluation(data):

    if not validate_evaluation(
        data
    ):

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
                    and is_safe_integer(
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
                if isinstance(data, dict)
                and is_safe_integer(
                    data.get(
                        "bytesProcessed"
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

    saved = get_run(
        data["runId"]
    )

    lineage_ok = False

    if saved is not None:

        old = saved["response"]

        if (
            old.get(
                "selectedTrialId"
            ) is not None
            and old.get(
                "datasetDigest"
            ) is not None
            and old.get(
                "reasonCodes"
            ) == []
            and old.get(
                "selectedTrialId"
            )
            == data["selectedTrialId"]
            and old.get(
                "datasetDigest"
            )
            == data["datasetDigest"]
        ):
            lineage_ok = True

    if not lineage_ok:
        reasons.append(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # TEST ROW VALIDATION
    # --------------------------------------------------------

    rows = data["rows"]

    bad_row = False

    for row in rows:

        if not validate_test_row(
            row
        ):
            bad_row = True
            break

    if bad_row:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if (
        len(rows) == 0
        or bad_row
    ):

        test_metric = None
        critical_slice_pass = False

    else:

        # Aggregate accuracy.
        correct = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = float(
            Decimal(correct)
            .__truediv__(
                Decimal(len(rows))
            ).quantize(
                Decimal("0.000000000001"),
                rounding=ROUND_HALF_UP
            )
        )

        if (
            test_metric
            < float(data["metricFloor"])
        ):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # Slice groups.
        # ----------------------------------------------------

        groups = {}

        for row in rows:

            groups.setdefault(
                row["slice"],
                []
            ).append(row)

        critical_slice_pass = True

        # ----------------------------------------------------
        # Required slices.
        # ----------------------------------------------------

        for name in utf8_sort(
            data["requiredSlices"].keys()
        ):

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

            slice_accuracy = float(
                Decimal(correct_slice)
                .__truediv__(
                    Decimal(len(group))
                ).quantize(
                    Decimal("0.000000000001"),
                    rounding=ROUND_HALF_UP
                )
            )

            if (
                slice_accuracy
                < float(
                    data[
                        "requiredSlices"
                    ][name]
                )
            ):

                reasons.append(
                    "SLICE_FLOOR:"
                    + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # Critical slice flag.
    # --------------------------------------------------------

    if not lineage_ok:
        critical_slice_pass = False

    if bad_row:
        critical_slice_pass = False

    if len(rows) == 0:
        critical_slice_pass = False

    # --------------------------------------------------------
    # COST.
    # --------------------------------------------------------

    if (
        data["bytesProcessed"]
        > data["maxBytes"]
    ):
        reasons.append(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # DECISION.
    # --------------------------------------------------------

    decision = "admit"

    if not lineage_ok:
        decision = "reject"

    if bad_row:
        decision = "reject"

    if len(rows) == 0:
        decision = "reject"

    if test_metric is not None:

        if (
            test_metric
            < float(data["metricFloor"])
        ):
            decision = "reject"

    else:
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
        "reasonCodes": sort_reason_codes(
            reasons
        )
    }


# ============================================================
# /bqml
# ============================================================

@app.post("/bqml")
def bqml():

    try:

        # Must be application/json.
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

        # Missing/unknown phase.
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

            # Invalid run ID is simply INVALID_INPUT.
            if not (
                isinstance(run_id, str)
                and run_id != ""
                and len(run_id) <= 128
            ):

                return jsonify(
                    build_selection(data)
                )

            # Fingerprint the entire selection input.
            fingerprint = sha256_compact_json(
                data
            )

            previous = get_run(
                run_id
            )

            if previous is not None:

                # Identical replay.
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

                # Same runId, different input.
                return jsonify({
                    "error":
                        "RUN_ID_CONFLICT"
                }), 409

            response = build_selection(
                data
            )

            # Persist complete response.
            put_run(
                run_id,
                fingerprint,
                response
            )

            return jsonify(
                response
            )

        # ====================================================
        # EVALUATE
        # ====================================================

        return jsonify(
            build_evaluation(data)
        )

    except Exception:
        # Prevent accidental 500s from malformed requests.
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
