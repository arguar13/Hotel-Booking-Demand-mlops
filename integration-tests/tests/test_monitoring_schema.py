"""Proves the two halves of the monitoring loop still agree on the schema.

`api/monitoring.py` owns the tables and writes to them; `core_ml/src/monitoring/
store.py` reads them back. They are separate Poetry projects that ship as
separate images, so nothing at import time can catch the day someone renames a
column on one side. The consequence of that divergence is not a test failure -
it is a CronJob that starts failing at 04:00, or worse, one that quietly returns
zero rows and reports "no drift" forever.

So this test takes the real SQL out of both source files - not a copy of it -
applies the writer's DDL to a throwaway Postgres container, writes through the
writer's own INSERT statements, and reads back through the reader's own SELECTs.
A rename on either side fails here, in CI, on the commit that caused it.

The SQL is extracted with `ast` rather than by importing the modules: importing
`api.monitoring` would drag in structlog, pybreaker and tenacity, and importing
`src.monitoring.store` would drag in mlflow and pandas - a large dependency
surface for this project to carry in order to read four string constants.
"""

import ast
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg2
import pytest
from psycopg2.extras import Json, execute_values
from testcontainers.community.postgres import PostgresContainer

REPO_ROOT = Path(__file__).resolve().parents[2]
API_MONITORING = REPO_ROOT / "api" / "monitoring.py"
CORE_ML_STORE = REPO_ROOT / "core_ml" / "src" / "monitoring" / "store.py"


def _string_constants(source_path: Path, names: set[str]) -> dict[str, str]:
    """Evaluate the named module-level string assignments, and nothing else.

    Each of the constants wanted here is either a plain string or an f-string
    whose only substitution is SCHEMA, so the expression is compiled in a
    namespace containing exactly that - no imports, no side effects, and no way
    for this helper to execute anything the module does at import time.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    namespace: dict[str, str] = {}
    wanted = names | {"SCHEMA"}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                expression = ast.Expression(body=node.value)
                ast.fix_missing_locations(expression)
                namespace[target.id] = eval(  # nosec B307 - compiled from this repo's own source
                    compile(expression, str(source_path), "eval"),
                    {"__builtins__": {}},
                    dict(namespace),
                )

    missing = names - namespace.keys()
    assert not missing, f"{source_path.name} no longer defines {missing}"
    return namespace


@pytest.fixture(scope="module")
def api_sql() -> dict[str, str]:
    return _string_constants(
        API_MONITORING,
        {"SCHEMA_DDL", "_INSERT_PREDICTION", "_UPSERT_LABELS", "_LABEL_VALUES_TEMPLATE"},
    )


@pytest.fixture(scope="module")
def monitor_sql() -> dict[str, str]:
    return _string_constants(
        CORE_ML_STORE, {"_SELECT_PREDICTIONS", "_SELECT_LABELLED", "_SELECT_COVERAGE"}
    )


@pytest.fixture(scope="module")
def connection(api_sql):
    """A real Postgres with the API's real schema applied."""
    with PostgresContainer("postgres:18.3") as postgres:
        conn = psycopg2.connect(
            host=postgres.get_container_host_ip(),
            port=postgres.get_exposed_port(5432),
            dbname=postgres.dbname,
            user=postgres.username,
            password=postgres.password,
        )
        with conn.cursor() as cur:
            cur.execute(api_sql["SCHEMA_DDL"])
        conn.commit()
        yield conn
        conn.close()


def _insert_prediction(conn, api_sql, *, predicted_at, segment, version="7", confidence=0.9):
    prediction_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            api_sql["_INSERT_PREDICTION"],
            (
                prediction_id,
                predicted_at,
                "HotelSegmentClassifier",
                "staging",
                version,
                segment,
                confidence,
                0.4,
                Json({"lead_time": 10, "adr": 98.5, "hotel": "Resort Hotel"}),
            ),
        )
    conn.commit()
    return prediction_id


def _label(conn, api_sql, rows) -> int:
    """Write labels exactly the way api/monitoring.py does, and report the count."""
    with conn.cursor() as cur:
        execute_values(
            cur,
            api_sql["_UPSERT_LABELS"],
            rows,
            template=api_sql["_LABEL_VALUES_TEMPLATE"],
        )
        written = cur.rowcount
    conn.commit()
    return max(int(written), 0)


def test_the_ddl_is_idempotent(connection, api_sql):
    """It is applied on every API pod start, so a second run must be a no-op."""
    with connection.cursor() as cur:
        cur.execute(api_sql["SCHEMA_DDL"])
    connection.commit()


def test_a_prediction_written_by_the_api_is_read_back_by_the_monitor(
    connection, api_sql, monitor_sql
):
    now = datetime.now(UTC)
    prediction_id = _insert_prediction(
        connection, api_sql, predicted_at=now - timedelta(hours=1), segment="Direct"
    )

    with connection.cursor() as cur:
        cur.execute(
            monitor_sql["_SELECT_PREDICTIONS"],
            {
                "window_start": now - timedelta(hours=24),
                "window_end": now + timedelta(minutes=1),
                "model_version": "7",
                "max_rows": 100,
            },
        )
        rows = cur.fetchall()

    assert any(str(row[0]) == prediction_id for row in rows)
    # The JSONB features are the entire input to data drift; losing their types
    # here would silently turn every numeric feature into a categorical one.
    features = next(row[6] for row in rows if str(row[0]) == prediction_id)
    assert features == {"lead_time": 10, "adr": 98.5, "hotel": "Resort Hotel"}


def test_the_concept_drift_join_returns_ground_truth(connection, api_sql, monitor_sql):
    now = datetime.now(UTC)
    prediction_id = _insert_prediction(
        connection, api_sql, predicted_at=now - timedelta(hours=2), segment="Direct"
    )

    _label(connection, api_sql, [(prediction_id, "Online TA", now, "reconciliation")])

    with connection.cursor() as cur:
        cur.execute(
            monitor_sql["_SELECT_LABELLED"],
            {
                "window_start": now - timedelta(hours=24),
                "window_end": now + timedelta(minutes=1),
                "model_version": "7",
                "max_rows": 100,
            },
        )
        rows = cur.fetchall()

    matched = [row for row in rows if str(row[0]) == prediction_id]
    assert matched, "the labelled prediction did not come back through the join"
    predicted, actual = matched[0][3], matched[0][7]
    assert (predicted, actual) == ("Direct", "Online TA")


def test_labels_are_idempotent_so_a_reconciliation_batch_can_be_retried(
    connection, api_sql, monitor_sql
):
    """A batch job is at-least-once by nature; re-posting must correct, not fail."""
    now = datetime.now(UTC)
    prediction_id = _insert_prediction(
        connection, api_sql, predicted_at=now - timedelta(hours=3), segment="Groups"
    )

    _label(connection, api_sql, [(prediction_id, "Groups", now, "reconciliation")])
    _label(connection, api_sql, [(prediction_id, "Corporate", now, "correction")])

    with connection.cursor() as cur:
        cur.execute(
            "SELECT actual_segment, label_source FROM monitoring.booking_labels "
            "WHERE prediction_id = %s",
            (prediction_id,),
        )
        assert cur.fetchall() == [("Corporate", "correction")]


def test_an_unknown_prediction_is_skipped_without_aborting_the_batch(connection, api_sql):
    """The behaviour the first end-to-end run forced into the design.

    Predictions are persisted asynchronously, so a caller can hold an id that has
    not reached the table yet - and with two API replicas, the pod serving
    /feedback may not be the one holding it. A plain INSERT hit the foreign key,
    and because a psycopg2 batch shares one transaction, that single row rejected
    all 250 in the batch. Joining against `predictions` instead skips the unknown
    row, keeps the rest, and reports through rowcount how many landed so the
    caller can re-post the difference.
    """
    now = datetime.now(UTC)
    known = _insert_prediction(
        connection, api_sql, predicted_at=now - timedelta(minutes=5), segment="Direct"
    )
    unknown = str(uuid.uuid4())

    written = _label(
        connection,
        api_sql,
        [
            (known, "Direct", now, "reconciliation"),
            (unknown, "Groups", now, "reconciliation"),
        ],
    )

    assert written == 1  # the known one landed; the unknown one was skipped, not fatal

    with connection.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM monitoring.booking_labels WHERE prediction_id = %s", (unknown,)
        )
        assert cur.fetchone()[0] == 0


def test_the_foreign_key_is_still_the_integrity_backstop(connection):
    """The join is control flow; the constraint is the guarantee.

    Skipping unknown ids is a deliberate tolerance in one statement, not a licence
    for orphan labels to exist - an orphan would inflate label coverage and
    quietly bias the accuracy the monitor reports.
    """
    with connection.cursor() as cur, pytest.raises(psycopg2.errors.ForeignKeyViolation):
        cur.execute(
            "INSERT INTO monitoring.booking_labels "
            "(prediction_id, actual_segment, labeled_at, label_source) "
            "VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), "Direct", datetime.now(UTC), "reconciliation"),
        )
    connection.rollback()


def test_coverage_counts_predictions_and_labels_over_the_window(connection, api_sql, monitor_sql):
    """The query the monitor reads before deciding whether it can decide at all."""
    now = datetime.now(UTC)
    version = f"cov-{uuid.uuid4().hex[:8]}"

    unlabelled = [
        _insert_prediction(
            connection,
            api_sql,
            predicted_at=now - timedelta(minutes=30),
            segment="Direct",
            version=version,
        )
        for _ in range(3)
    ]
    labelled = _insert_prediction(
        connection,
        api_sql,
        predicted_at=now - timedelta(minutes=20),
        segment="Direct",
        version=version,
    )
    _label(connection, api_sql, [(labelled, "Direct", now, "reconciliation")])

    with connection.cursor() as cur:
        cur.execute(
            monitor_sql["_SELECT_COVERAGE"],
            {
                "window_start": now - timedelta(hours=1),
                "window_end": now + timedelta(minutes=1),
                "model_version": version,
            },
        )
        n_predictions, n_labelled, first_seen, last_seen = cur.fetchone()

    assert n_predictions == len(unlabelled) + 1
    assert n_labelled == 1
    assert first_seen is not None and last_seen is not None


def test_the_window_is_half_open_so_adjacent_windows_do_not_double_count(
    connection, api_sql, monitor_sql
):
    """[start, end) - a prediction on the boundary belongs to exactly one window.

    Counting it in both would let a single alerting event satisfy the
    consecutive-windows hysteresis on its own.
    """
    boundary = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=10)
    version = f"edge-{uuid.uuid4().hex[:8]}"
    _insert_prediction(
        connection, api_sql, predicted_at=boundary, segment="Direct", version=version
    )

    def count(start, end):
        with connection.cursor() as cur:
            cur.execute(
                monitor_sql["_SELECT_COVERAGE"],
                {"window_start": start, "window_end": end, "model_version": version},
            )
            return cur.fetchone()[0]

    earlier = count(boundary - timedelta(hours=1), boundary)
    later = count(boundary, boundary + timedelta(hours=1))

    assert (earlier, later) == (0, 1)


def test_features_survive_a_json_round_trip_with_nulls(connection, api_sql):
    """`jsonable_features` maps NaN to null; the column has to accept that."""
    prediction_id = str(uuid.uuid4())
    payload = {"adr": None, "lead_time": 0, "country": "PRT"}
    with connection.cursor() as cur:
        cur.execute(
            api_sql["_INSERT_PREDICTION"],
            (
                prediction_id,
                datetime.now(UTC),
                "HotelSegmentClassifier",
                "staging",
                "7",
                "Direct",
                None,
                None,
                Json(payload),
            ),
        )
        cur.execute(
            "SELECT features FROM monitoring.predictions WHERE prediction_id = %s",
            (prediction_id,),
        )
        stored = cur.fetchone()[0]
    connection.commit()

    assert stored == json.loads(json.dumps(payload))
