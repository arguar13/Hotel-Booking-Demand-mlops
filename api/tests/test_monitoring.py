"""The inference sink, tested at its two dangerous edges.

Edge one: it must be impossible for this component to hurt serving. Every test
here that simulates a broken database asserts that nothing propagates back to
the caller - because the moment logging a prediction can fail a prediction, the
right move operationally is to turn logging off, and then there is no drift
monitoring at all.

Edge two: it must not lie about what it stored. Silently dropping rows would let
the drift monitor compute a confident verdict over a biased sample, which is
strictly worse than reporting no verdict - so drops are counted, surfaced on
/health, and asserted here.
"""

import queue
from unittest.mock import MagicMock

import psycopg2
import pybreaker
import pytest
from fastapi.testclient import TestClient

import main
import monitoring
from monitoring import InferenceLogger, jsonable_features, new_prediction_id

client = TestClient(main.app)


@pytest.fixture
def logger(monkeypatch):
    """A logger with a DSN (so it is 'enabled') and a fake connection behind it."""
    instance = InferenceLogger(dsn="host=stub dbname=stub")
    instance._schema_ready = True

    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(instance, "acquire", lambda: connection)
    monkeypatch.setattr(instance, "release", lambda conn, close=False: None)

    executed: list[list[tuple]] = []

    def _capture_batch(cur, sql, rows, page_size=None):
        executed.append(list(rows))

    def _capture_values(cur, sql, rows, template=None, page_size=None):
        captured = list(rows)
        executed.append(captured)
        # The real statement joins against `predictions` and reports how many
        # rows landed; the fake stands in for "every id was already durable".
        cur.rowcount = len(captured)

    monkeypatch.setattr(monitoring.extras, "execute_batch", _capture_batch)
    monkeypatch.setattr(monitoring.extras, "execute_values", _capture_values)
    instance.executed = executed  # type: ignore[attr-defined]
    instance.cursor = cursor  # type: ignore[attr-defined]
    return instance


def _record(instance: InferenceLogger, segment: str = "Direct") -> None:
    instance.record_prediction(
        prediction_id=new_prediction_id(),
        model_name="HotelSegmentClassifier",
        model_alias="staging",
        model_version="7",
        predicted_segment=segment,
        confidence=0.9,
        margin=0.4,
        features={"lead_time": 10, "adr": 98.0},
    )


# --- disabled mode ---------------------------------------------------------


def test_logger_is_disabled_without_database_configuration():
    """No DB_HOST means no sink, and that must be inert rather than broken."""
    instance = InferenceLogger(dsn=None)

    assert instance.enabled is False
    _record(instance)  # must not raise
    assert instance.flush() == 0
    assert instance.stats()["enabled"] is False


def test_recording_labels_while_disabled_raises_rather_than_pretending():
    """Predictions may be dropped silently; labels may not.

    A lost prediction costs a row of a statistical sample. A lost label makes
    the accuracy computed from the remaining ones wrong, so the caller has to
    find out.
    """
    instance = InferenceLogger(dsn=None)

    with pytest.raises(RuntimeError):
        instance.record_labels([("id", "Direct", "reconciliation")])


# --- the write path --------------------------------------------------------


def test_flush_writes_the_queued_batch(logger):
    for _ in range(3):
        _record(logger)

    assert logger.flush() == 3
    assert len(logger.executed[0]) == 3
    assert logger.stats()["rows_written"] == 3
    assert logger.stats()["queue_depth"] == 0


def test_flush_is_a_noop_on_an_empty_queue(logger):
    assert logger.flush() == 0
    assert logger.executed == []


def test_a_database_failure_never_propagates_to_the_caller(logger, monkeypatch):
    """The property the whole design exists to guarantee."""
    monkeypatch.setattr(
        monitoring.extras,
        "execute_batch",
        MagicMock(side_effect=psycopg2.OperationalError("connection refused")),
    )
    _record(logger)

    assert logger.flush() == 0  # swallowed, not raised
    assert logger.stats()["rows_dropped"] == 1


def test_the_breaker_opens_and_then_stops_calling_a_dead_database(logger, monkeypatch):
    """Fail fast instead of paying a connection timeout on every flush."""
    failing = MagicMock(side_effect=psycopg2.OperationalError("down"))
    monkeypatch.setattr(monitoring.extras, "execute_batch", failing)

    for _ in range(logger.breaker.fail_max):
        _record(logger)
        logger.flush()

    assert logger.breaker.current_state == "open"

    calls_before = failing.call_count
    _record(logger)
    assert logger.flush() == 0
    assert failing.call_count == calls_before  # short-circuited by the breaker


def test_the_queue_is_bounded_and_drops_the_oldest_record(logger, monkeypatch):
    """Bounded memory under backpressure, with the drop counted rather than hidden.

    Without this the queue grows until the kubelet OOM-kills the pod, which
    takes serving down for the sake of observability data - exactly the
    inversion the design forbids.
    """
    monkeypatch.setattr(logger, "_queue", queue.Queue(maxsize=2))

    _record(logger, "first")
    _record(logger, "second")
    _record(logger, "third")

    assert logger.stats()["queue_depth"] == 2
    assert logger.stats()["rows_dropped"] == 1

    logger.flush()
    segments = [row[5] for row in logger.executed[0]]
    assert segments == ["second", "third"]  # the oldest was shed, the newest kept


def test_start_never_raises_when_the_database_is_unreachable(monkeypatch):
    """A failed sink must not fail startup - /ready gates on the model, not this."""
    instance = InferenceLogger(dsn="host=stub")
    monkeypatch.setattr(
        instance, "_ensure_schema", MagicMock(side_effect=psycopg2.OperationalError("nope"))
    )

    instance.start()  # must not raise

    assert instance.enabled is True


def test_labels_are_written_through_the_breaker(logger):
    accepted = logger.record_labels(
        [("id-1", "Direct", "reconciliation"), ("id-2", "Online TA", "reconciliation")]
    )

    assert accepted == 2
    assert len(logger.executed[0]) == 2


def test_label_write_failure_is_raised_not_swallowed(logger, monkeypatch):
    monkeypatch.setattr(
        monitoring.extras,
        "execute_values",
        MagicMock(side_effect=psycopg2.OperationalError("down")),
    )

    with pytest.raises((psycopg2.Error, pybreaker.CircuitBreakerError)):
        logger.record_labels([("id-1", "Direct", "reconciliation")])


# --- serialization ---------------------------------------------------------


def test_nan_and_infinity_are_normalised_to_null():
    """JSONB rejects the bare NaN token json.dumps emits, and one unrepresentable
    value must not fail the whole batch it happens to share."""
    cleaned = jsonable_features(
        {"adr": float("nan"), "lead_time": float("inf"), "adults": 2, "hotel": "Resort Hotel"}
    )

    assert cleaned == {"adr": None, "lead_time": None, "adults": 2, "hotel": "Resort Hotel"}


# --- the API surface -------------------------------------------------------


@pytest.fixture
def loaded_model(monkeypatch):
    """A stand-in classifier that exposes predict_proba, as RandomForest does."""
    model = MagicMock()
    model.classes_ = ["Direct", "Groups", "Online TA"]
    model.predict_proba.return_value = [[0.7, 0.1, 0.2]]
    model.predict.return_value = ["Direct"]
    monkeypatch.setattr(main, "model", model)
    monkeypatch.setattr(main, "model_version", "7")
    return model


PAYLOAD = {
    "hotel": "Resort Hotel",
    "lead_time": 10,
    "arrival_date_year": 2017,
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 3,
    "adults": 2,
    "meal": "BB",
    "country": "PRT",
    "distribution_channel": "Direct",
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "adr": 98.0,
    "month": 7,
}


def test_predict_returns_the_correlation_id_and_confidence(loaded_model):
    """Without prediction_id nothing can ever be joined to ground truth, and
    without confidence there is no unsupervised drift signal at all."""
    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_market_segment"] == "Direct"
    assert body["prediction_id"]
    assert body["model_version"] == "7"
    assert body["confidence"] == pytest.approx(0.7)
    # 0.7 (top) - 0.2 (runner-up): the gap a single confidence figure hides.
    assert body["margin"] == pytest.approx(0.5)


def test_predict_degrades_to_a_bare_label_without_predict_proba(monkeypatch):
    """An estimator with no probabilities must lose the signal, not the endpoint."""
    model = MagicMock(spec=["predict"])
    model.predict.return_value = ["Groups"]
    monkeypatch.setattr(main, "model", model)

    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json()["predicted_market_segment"] == "Groups"
    assert response.json()["confidence"] is None


def test_feedback_is_503_when_there_is_no_prediction_log(monkeypatch):
    monkeypatch.setattr(monitoring.inference_logger, "_dsn", None)

    response = client.post(
        "/feedback",
        json=[{"prediction_id": new_prediction_id(), "actual_market_segment": "Direct"}],
    )

    assert response.status_code == 503


def test_feedback_accepts_an_empty_batch_without_touching_the_database():
    """A reconciliation job with nothing to report is not an error."""
    assert client.post("/feedback", json=[]).json() == {"accepted": 0, "skipped": 0}


def test_labels_for_predictions_not_yet_durable_are_skipped_not_rejected(logger):
    """The race the end-to-end run exposed: /predict persists asynchronously, so
    a fast reconciliation can quote an id that has not reached the table yet.

    Those rows must be reported as skipped - a retryable, expected outcome -
    rather than aborting the whole batch on a foreign key, which is what a plain
    INSERT did and what made a 250-row batch fail because of one row.
    """

    def _partial(cur, sql, rows, template=None, page_size=None):
        cur.rowcount = 1  # only one of the two predictions was durable

    logger.cursor.rowcount = 1
    import monitoring as monitoring_module

    monitoring_module.extras.execute_values = _partial

    accepted = logger.record_labels(
        [("id-1", "Direct", "reconciliation"), ("id-2", "Groups", "reconciliation")]
    )

    assert accepted == 1


def test_health_surfaces_the_sink_state(loaded_model):
    """A degraded sink must be visible without reading logs - rows_dropped
    climbing is the warning that tomorrow's drift report will be computed on an
    unrepresentative sample."""
    body = client.get("/health").json()

    assert "inference_logging" in body
    assert set(body["inference_logging"]) >= {
        "enabled",
        "queue_depth",
        "rows_written",
        "rows_dropped",
        "circuit_breaker_state",
    }
    assert body["model_version"] == "7"
