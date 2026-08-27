from unittest.mock import MagicMock

import pybreaker
import pytest
from fastapi.testclient import TestClient
from kafka.errors import KafkaError

import main

client = TestClient(main.app)


def test_health_endpoint():
    """
    /health must always return 200, regardless of whether a model is
    currently loaded - it reports process liveness, not model readiness.
    """
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_ready_endpoint_is_503_without_a_model(monkeypatch):
    """
    /ready gates traffic, so it must fail while there is no model.

    Regression test: readiness used to point at /health, which is 200
    unconditionally. A replica that came up while MLflow was still starting
    therefore joined the Service and answered every /predict with a 503, and
    because /health stayed 200 Kubernetes never restarted it.
    """
    monkeypatch.setattr(main, "model", None)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_ready_endpoint_is_200_once_a_model_is_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", MagicMock())

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_endpoint_structure():
    """
    Verifica que el endpoint /predict responda con el formato correcto
    al enviar un payload válido.
    """
    payload = {
        "hotel": "Resort Hotel",
        "lead_time": 342,
        "arrival_date_year": 2015,
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "C",
        "assigned_room_type": "C",
        "booking_changes": 3,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 0.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "month": 7,
        "is_canceled": 0,
        "agent": 9.0,
        "company": 0.0,
    }

    response = client.post("/predict", json=payload)

    # Si el modelo está cargado debe responder 200, si no está cargado 503
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "predicted_market_segment" in data


@pytest.fixture(autouse=True)
def _reset_kafka_breaker():
    """Every test gets a closed breaker, regardless of what earlier tests did to it."""
    main._kafka_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)
    yield
    main._kafka_producer = None


def test_kafka_publish_never_raises_even_when_kafka_is_down():
    """A downed Kafka must never surface as a failure of the prediction request."""
    main._kafka_producer = MagicMock()
    main._kafka_producer.send.side_effect = KafkaError("boom")

    features = main.BookingFeatures(
        hotel="Resort Hotel",
        lead_time=1,
        arrival_date_year=2015,
        arrival_date_week_number=27,
        arrival_date_day_of_month=1,
        stays_in_weekend_nights=0,
        stays_in_week_nights=1,
        adults=2,
        meal="BB",
        country="PRT",
        distribution_channel="Direct",
        reserved_room_type="C",
        assigned_room_type="C",
        deposit_type="No Deposit",
        customer_type="Transient",
        adr=100.0,
        month=7,
    )

    main._publish_prediction_event(features, "Direct")  # must not raise


def test_kafka_circuit_breaker_opens_after_repeated_failures():
    """After fail_max consecutive failures, the breaker must stop calling Kafka
    at all - the whole point of a breaker is to fail fast instead of retrying
    a dependency that has already proven it's down.
    """
    main._kafka_producer = MagicMock()
    main._kafka_producer.send.side_effect = KafkaError("boom")

    features = main.BookingFeatures(
        hotel="Resort Hotel",
        lead_time=1,
        arrival_date_year=2015,
        arrival_date_week_number=27,
        arrival_date_day_of_month=1,
        stays_in_weekend_nights=0,
        stays_in_week_nights=1,
        adults=2,
        meal="BB",
        country="PRT",
        distribution_channel="Direct",
        reserved_room_type="C",
        assigned_room_type="C",
        deposit_type="No Deposit",
        customer_type="Transient",
        adr=100.0,
        month=7,
    )

    for _ in range(main._kafka_breaker.fail_max):
        main._publish_prediction_event(features, "Direct")
    assert main._kafka_breaker.current_state == "open"

    calls_before = main._kafka_producer.send.call_count
    main._publish_prediction_event(features, "Direct")  # breaker should short-circuit this
    assert main._kafka_producer.send.call_count == calls_before
