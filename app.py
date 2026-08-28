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


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(
        tempfile.gettempdir(),
        "bqml_state.sqlite3"
    )
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
# DATABASE / STATE
# ============================================================

def get_db():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=10
    )

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

    try:
        response = json.loads(row[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    return {
        "fingerprint": row[0],
        "response": response
    }


def save_run(run_id, fingerprint, response):
    """
    Persist the complete selection response.

    If another request won a race and inserted the same run_id,
    the caller can reload the stored record and perform the
    normal replay/conflict logic.
    """

    serialized = json.dumps(
        response,
        ensure_ascii=False,
        separators=(",", ":")
    )

    conn = get_db()

    try:
        conn.execute(
            """
            INSERT INTO runs
            (run_id, fingerprint, response)
            VALUES (?, ?, ?)
            """,
            (
                run_id,
                fingerprint,
                serialized
            )
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:
        conn.rollback()
        return False

    finally:
        conn.close()


# ============================================================
# BASIC VALIDATION
# ============================================================

def safe_integer(value):
    """
    Non-negative JavaScript-safe integer.
    bool is deliberately rejected because bool is a subclass
    of int in Python.
    """

    return (
        type(value) is int
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def positive_integer(value):
    """
    Positive integer.

    The prompt specifically limits versions/trial IDs to
    safe integers, but describes numTrialsLimit simply as
    a positive integer, so this intentionally does not impose
    MAX_SAFE_INTEGER on numTrialsLimit.
    """

    return (
        type(value) is int
        and value > 0
    )


def finite_number(value):
    """
    Finite JSON number.

    bool is not considered a number here.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(
        value,
        (int, float)
    ):
        return False

    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
        OverflowError
    ):
        return False


def utf8_key(value):
    return value.encode("utf-8")


def utf8_sorted(values):
    return sorted(
        values,
        key=utf8_key
    )


def sorted_codes(codes):
    """
    Deduplicate reason codes and sort by UTF-8 bytes.
    """

    return sorted(
        set(codes),
        key=utf8_key
    )


# ============================================================
# TIMESTAMP HANDLING
# ============================================================

def parse_timestamp(value):
    """
    Valid forms:

      YYYY-MM-DDTHH:mm:ssZ
      YYYY-MM-DDTHH:mm:ss.sZ
      YYYY-MM-DDTHH:mm:ss.ssZ
      YYYY-MM-DDTHH:mm:ss.sssZ

    or equivalent forms using ±HH:mm.

    Returns a UTC-aware datetime or None.
    """

    if not isinstance(
        value,
        str
    ):
        return None

    if not TIMESTAMP_RE.fullmatch(
        value
    ):
        return None

    try:
        text = value

        if text.endswith("Z"):
            text = (
                text[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            text
        )

        if dt.tzinfo is None:
            return None

        return dt.astimezone(
            timezone.utc
        )

    except (
        TypeError,
        ValueError,
        OverflowError
    ):
        return None


# ============================================================
# JSON / DIGESTS
# ============================================================

def compact_json(value):
    """
    Compact JSON using UTF-8 directly.

    No ASCII escaping is used because the digest is over the
    compact JSON representation with the supplied Unicode data.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def dataset_digest(
    train_ids,
    eval_ids,
    feature_names
):
    """
    EXACT required object shape and key order:

    {
      "trainRowIds": ...,
      "evalRowIds": ...,
      "featureNames": ...
    }
    """

    value = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    encoded = compact_json(
        value
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def request_fingerprint(data):
    """
    Canonical fingerprint of the complete selection request.

    JSON object key ordering is normalized so the same logical
    request does not become a conflict merely because object
    members were supplied in a different order.
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
    if not isinstance(
        feature,
        dict
    ):
        return False

    # "value" is DATA.
    #
    # Do not impose a type restriction on it.
    # Strings, numbers, booleans, null, objects, arrays, etc.
    # are all allowed as feature data.
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
    if not isinstance(
        row,
        dict
    ):
        return False

    required_keys = (
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    )

    for key in required_keys:
        if key not in row:
            return False

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if not isinstance(
        row["id"],
        str
    ):
        return False

    if row["id"] == "":
        return False

    # --------------------------------------------------------
    # ENTITY
    # --------------------------------------------------------

    if not isinstance(
        row["entity"],
        str
    ):
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
    # POINT-IN-TIME ROW SAFETY
    # --------------------------------------------------------
    #
    # An event occurring after the prediction point is invalid.
    # Equality is allowed.
    #

    if event_time > prediction_time:
        return False

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
    if not isinstance(
        rows,
        list
    ):
        return False

    # Selection rows must be non-empty.
    if len(rows) == 0:
        return False

    seen_ids = set()

    for row in rows:

        if not valid_selection_row(
            row
        ):
            return False

        row_id = row["id"]

        # IDs are unique within the supplied array.
        if row_id in seen_ids:
            return False

        seen_ids.add(
            row_id
        )

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(rows):
    """
    Deduplicate by:

        [entity, UTC(eventTime)]

    Winner:

        highest version

    Exact version tie:

        UTF-8-byte-smallest ID
    """

    groups = {}

    for row in rows:

        event_utc = parse_timestamp(
            row["eventTime"]
        )

        # valid_selection_row() has already checked this.
        if event_utc is None:
            continue

        key = (
            row["entity"],
            event_utc
        )

        old = groups.get(
            key
        )

        if old is None:
            groups[key] = row
            continue

        # Highest version wins.
        if (
            row["version"]
            > old["version"]
        ):
            groups[key] = row
            continue

        # Equal version -> smallest UTF-8 ID.
        if (
            row["version"]
            == old["version"]
        ):

            if (
                utf8_key(row["id"])
                <
                utf8_key(old["id"])
            ):
                groups[key] = row

    return list(
        groups.values()
    )


# ============================================================
# POINT-IN-TIME / SHARED FEATURES
# ============================================================

def eligible_features(
    retained_rows,
    forbidden_features
):
    """
    A feature is eligible iff:

    1. It exists in EVERY retained row.
    2. It is not forbidden.
    3. For EVERY retained row:
           availableAt <= predictionTime

    Returned names are sorted by UTF-8 bytes.
    """

    if len(
        retained_rows
    ) == 0:
        return []

    forbidden = set(
        forbidden_features
    )

    # --------------------------------------------------------
    # Start with the first row's features.
    # --------------------------------------------------------

    shared = set(
        retained_rows[0][
            "features"
        ].keys()
    )

    # --------------------------------------------------------
    # Feature must occur in EVERY retained row.
    # --------------------------------------------------------

    for row in retained_rows[1:]:

        shared.intersection_update(
            row["features"].keys()
        )

    # --------------------------------------------------------
    # Forbidden features are never eligible.
    # --------------------------------------------------------

    shared.difference_update(
        forbidden
    )

    result = []

    # --------------------------------------------------------
    # Point-in-time test.
    # --------------------------------------------------------

    for feature_name in shared:

        feature_is_eligible = True

        for row in retained_rows:

            available_at = parse_timestamp(
                row["features"][
                    feature_name
                ]["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            # Defensive validation.
            if (
                available_at is None
                or prediction_time is None
            ):
                feature_is_eligible = False
                break

            # IMPORTANT:
            # availableAt equal to predictionTime is valid.
            if available_at > prediction_time:
                feature_is_eligible = False
                break

        if feature_is_eligible:
            result.append(
                feature_name
            )

    return utf8_sorted(
        result
    )


# ============================================================
# TRIAL VALIDATION
# ============================================================

def valid_trial(trial):
    if not isinstance(
        trial,
        dict
    ):
        return False

    required_keys = (
        "trialId",
        "status",
        "evalMetric"
    )

    for key in required_keys:
        if key not in trial:
            return False

    # Trial ID must be a non-negative safe integer.
    if not safe_integer(
        trial["trialId"]
    ):
        return False

    if trial["status"] not in (
        "SUCCEEDED",
        "FAILED"
    ):
        return False

    # evalMetric is only required to be finite when the trial
    # is SUCCEEDED and therefore eligible.
    #
    # Do not impose an artificial [0,1] restriction because
    # the contract does not specify one.

    return True


def valid_trials(trials):
    if not isinstance(
        trials,
        list
    ):
        return False

    seen_ids = set()

    for trial in trials:

        if not valid_trial(
            trial
        ):
            return False

        trial_id = trial[
            "trialId"
        ]

        if trial_id in seen_ids:
            return False

        seen_ids.add(
            trial_id
        )

    return True


# ============================================================
# TRIAL SELECTION
# ============================================================

def select_trial(trials):
    """
    Eligible:

      status == SUCCEEDED
      AND evalMetric is finite

    Winner:

      maximum evalMetric

    Exact metric tie:

      smallest integer trialId
    """

    eligible = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        if not finite_number(
            trial["evalMetric"]
        ):
            continue

        eligible.append(
            trial
        )

    if not eligible:
        return None

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

    if data.get(
        "phase"
    ) != "select":
        return False

    # --------------------------------------------------------
    # runId
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # forbiddenFeatures
    # --------------------------------------------------------

    forbidden = data.get(
        "forbiddenFeatures"
    )

    if not isinstance(
        forbidden,
        list
    ):
        return False

    for feature_name in forbidden:

        if not isinstance(
            feature_name,
            str
        ):
            return False

    # --------------------------------------------------------
    # numTrialsLimit
    # --------------------------------------------------------

    if not positive_integer(
        data.get(
            "numTrialsLimit"
        )
    ):
        return False

    # --------------------------------------------------------
    # rows
    # --------------------------------------------------------

    if not valid_selection_rows(
        data.get(
            "rows"
        )
    ):
        return False

    # --------------------------------------------------------
    # trials
    # --------------------------------------------------------

    if not valid_trials(
        data.get(
            "trials"
        )
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

    if not valid_selection(
        data
    ):

        return {
            "runId": (
                data.get(
                    "runId",
                    ""
                )
                if isinstance(
                    data,
                    dict
                )
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
        >
        data["numTrialsLimit"]
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
    #
    # This is deliberately BEFORE the TRAIN/EVAL split.
    #

    retained = deduplicate(
        data["rows"]
    )

    # --------------------------------------------------------
    # SPLIT AFTER DEDUPLICATION
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
    # SHARED POINT-IN-TIME-SAFE FEATURES
    # --------------------------------------------------------

    feature_names = eligible_features(
        retained,
        data["forbiddenFeatures"]
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------
    #
    # Only phase="select" rows reach this function.
    #
    # Final-test rows supplied during phase="evaluate" are
    # never inspected here.
    #

    selected = select_trial(
        data["trials"]
    )

    # --------------------------------------------------------
    # NO SUCCESSFUL FINITE TRIAL
    # --------------------------------------------------------

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
    # FREEZE DATASET LINEAGE
    # --------------------------------------------------------

    digest = dataset_digest(
        train_ids,
        eval_ids,
        feature_names
    )

    # --------------------------------------------------------
    # SUCCESSFUL SELECTION
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
# FINAL TEST ROW VALIDATION
# ============================================================

def valid_test_row(row):
    if not isinstance(
        row,
        dict
    ):
        return False

    # EXACT integer type.
    #
    # Do not use "value in (0, 1)" alone because Python treats
    # True as equal to 1 and False as equal to 0.
    if type(
        row.get("label")
    ) is not int:
        return False

    if row["label"] not in (
        0,
        1
    ):
        return False

    if type(
        row.get("prediction")
    ) is not int:
        return False

    if row["prediction"] not in (
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
# EVALUATION INPUT VALIDATION
# ============================================================

def valid_evaluation(data):
    if not isinstance(
        data,
        dict
    ):
        return False

    if data.get(
        "phase"
    ) != "evaluate":
        return False

    # --------------------------------------------------------
    # runId
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # selectedTrialId
    # --------------------------------------------------------

    if not safe_integer(
        data.get(
            "selectedTrialId"
        )
    ):
        return False

    # --------------------------------------------------------
    # datasetDigest
    # --------------------------------------------------------

    digest = data.get(
        "datasetDigest"
    )

    if not isinstance(
        digest,
        str
    ):
        return False

    if not DIGEST_RE.fullmatch(
        digest
    ):
        return False

    # --------------------------------------------------------
    # metricFloor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # requiredSlices
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # test rows
    # --------------------------------------------------------

    rows = data.get(
        "rows"
    )

    if not isinstance(
        rows,
        list
    ):
        return False

    # Empty rows are allowed at input-validation level.
    # They produce testMetric=null and reject later.

    # --------------------------------------------------------
    # bytes
    # --------------------------------------------------------

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
# ACCURACY ROUNDING
# ============================================================

def round12(value):
    """
    Round to exactly 12 decimal places using decimal
    half-up rounding, then return JSON-friendly float.
    """

    return float(
        Decimal(
            str(value)
        ).quantize(
            Decimal(
                "0.000000000001"
            ),
            rounding=ROUND_HALF_UP
        )
    )


# ============================================================
# LINEAGE
# ============================================================

def valid_stored_selection(response):
    if not isinstance(
        response,
        dict
    ):
        return False

    # Successful selection has no reason codes.
    if response.get(
        "reasonCodes"
    ) != []:
        return False

    # Selected trial must be a safe non-negative integer.
    if not safe_integer(
        response.get(
            "selectedTrialId"
        )
    ):
        return False

    # Digest must be a valid lowercase SHA-256 hex string.
    digest = response.get(
        "datasetDigest"
    )

    if not isinstance(
        digest,
        str
    ):
        return False

    if not DIGEST_RE.fullmatch(
        digest
    ):
        return False

    # Dataset arrays must have the expected types.
    if not isinstance(
        response.get(
            "trainRowIds"
        ),
        list
    ):
        return False

    if not isinstance(
        response.get(
            "evalRowIds"
        ),
        list
    ):
        return False

    if not isinstance(
        response.get(
            "featureNames"
        ),
        list
    ):
        return False

    return True


def lineage_matches(
    data,
    stored
):
    if stored is None:
        return False

    response = stored.get(
        "response"
    )

    if not valid_stored_selection(
        response
    ):
        return False

    if (
        response["selectedTrialId"]
        != data["selectedTrialId"]
    ):
        return False

    if (
        response["datasetDigest"]
        != data["datasetDigest"]
    ):
        return False

    return True


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def perform_evaluation(data):

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    if not valid_evaluation(
        data
    ):

        return {
            "runId": (
                data.get(
                    "runId",
                    ""
                )
                if isinstance(
                    data,
                    dict
                )
                else ""
            ),
            "selectedTrialId": (
                data.get(
                    "selectedTrialId"
                )
                if (
                    isinstance(
                        data,
                        dict
                    )
                    and safe_integer(
                        data.get(
                            "selectedTrialId"
                        )
                    )
                )
                else None
            ),
            "datasetDigest": (
                data.get(
                    "datasetDigest"
                )
                if isinstance(
                    data,
                    dict
                )
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
                    isinstance(
                        data,
                        dict
                    )
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
    # LINEAGE GATE
    # --------------------------------------------------------

    stored = load_run(
        data["runId"]
    )

    lineage_valid = lineage_matches(
        data,
        stored
    )

    if not lineage_valid:
        reasons.append(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # FINAL TEST ROW VALIDATION
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
    # EMPTY / INVALID ROW GATE
    # --------------------------------------------------------
    #
    # If rows are empty OR any row is invalid:
    #
    #   testMetric = null
    #   aggregate check is skipped
    #   required-slice checks are skipped
    #
    # Lineage and byte gates still apply.
    #

    if (
        len(rows) == 0
        or invalid_test_row
    ):

        test_metric = None

        critical_slice_pass = False

    else:

        # ====================================================
        # AGGREGATE ACCURACY
        # ====================================================

        correct = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = round12(
            correct / len(rows)
        )

        aggregate_pass = (
            test_metric
            >= float(
                data["metricFloor"]
            )
        )

        if not aggregate_pass:
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ====================================================
        # SLICE GROUPING
        # ====================================================

        slices = {}

        for row in rows:

            slices.setdefault(
                row["slice"],
                []
            ).append(row)

        # This gate summarizes ONLY required slices.
        critical_slice_pass = True

        # ====================================================
        # REQUIRED SLICE CHECKS
        # ====================================================

        required_slice_names = utf8_sorted(
            data[
                "requiredSlices"
            ].keys()
        )

        for name in required_slice_names:

            # Required slice must exist.
            if name not in slices:

                reasons.append(
                    "MISSING_SLICE:"
                    + name
                )

                critical_slice_pass = False

                continue

            slice_rows = slices[
                name
            ]

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

            slice_floor = float(
                data[
                    "requiredSlices"
                ][name]
            )

            # Inclusive floor.
            if slice_metric < slice_floor:

                reasons.append(
                    "SLICE_FLOOR:"
                    + name
                )

                critical_slice_pass = False

    # --------------------------------------------------------
    # criticalSlicePass DOES NOT summarize:
    #
    #   aggregate floor
    #   byte limit
    #
    # It only represents the required-slice portion plus the
    # conditions explicitly stated by the contract.
    # --------------------------------------------------------

    if not lineage_valid:
        critical_slice_pass = False

    if invalid_test_row:
        critical_slice_pass = False

    if len(rows) == 0:
        critical_slice_pass = False

    # --------------------------------------------------------
    # BYTE GATE
    # --------------------------------------------------------

    bytes_pass = (
        data["bytesProcessed"]
        <= data["maxBytes"]
    )

    if not bytes_pass:
        reasons.append(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # AGGREGATE GATE
    # --------------------------------------------------------

    aggregate_pass = (
        test_metric is not None
        and
        test_metric
        >= float(
            data["metricFloor"]
        )
    )

    # --------------------------------------------------------
    # ROW GATE
    # --------------------------------------------------------

    rows_pass = (
        len(rows) > 0
        and not invalid_test_row
    )

    # --------------------------------------------------------
    # FINAL ADMISSION
    # --------------------------------------------------------
    #
    # ALL independent gates must pass:
    #
    #   lineage
    #   every test row valid / non-empty
    #   aggregate floor
    #   every required slice
    #   byte limit
    #

    decision = "admit"

    if not lineage_valid:
        decision = "reject"

    if not rows_pass:
        decision = "reject"

    if not aggregate_pass:
        decision = "reject"

    if not critical_slice_pass:
        decision = "reject"

    if not bytes_pass:
        decision = "reject"

    # --------------------------------------------------------
    # EXACT EVALUATION RESPONSE
    # --------------------------------------------------------

    return {
        "runId": data[
            "runId"
        ],
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

    # --------------------------------------------------------
    # APPLICATION/JSON
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # UNKNOWN / MISSING PHASE
    # --------------------------------------------------------

    if phase not in (
        "select",
        "evaluate"
    ):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # ========================================================
    # SELECTION PHASE
    # ========================================================

    if phase == "select":

        run_id = data.get(
            "runId"
        )

        # ----------------------------------------------------
        # Malformed runId:
        #
        # Return normal selection INVALID_INPUT response.
        # Do not persist it.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Existing run?
        # ----------------------------------------------------

        previous = load_run(
            run_id
        )

        if previous is not None:

            # Identical replay -> EXACT stored response.
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

            # Same ID + different selection input.
            return jsonify({
                "error":
                    "RUN_ID_CONFLICT"
            }), 409

        # ----------------------------------------------------
        # Calculate response.
        # ----------------------------------------------------

        response = perform_selection(
            data
        )

        # ----------------------------------------------------
        # Persist complete response.
        # ----------------------------------------------------

        inserted = save_run(
            run_id,
            fingerprint,
            response
        )

        # ----------------------------------------------------
        # Handle concurrent insertion.
        # ----------------------------------------------------

        if not inserted:

            previous = load_run(
                run_id
            )

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

                return jsonify({
                    "error":
                        "RUN_ID_CONFLICT"
                }), 409

            # Extremely unusual state failure.
            return jsonify({
                "error": "INVALID_INPUT"
            }), 400

        return jsonify(
            response
        )

    # ========================================================
    # EVALUATION PHASE
    # ========================================================

    return jsonify(
        perform_evaluation(
            data
        )
    )


# ============================================================
# HEALTH / ROOT
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():
    return "BQML service running"


# ============================================================
# LOCAL DEVELOPMENT
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
