import asyncio
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

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


PAYLOAD = {
    "hotel": "Resort Hotel",
    "lead_time": 342,
    "month": 7,
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0.0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "C",
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 98.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "agent": 9.0,
    "company": None,
}


def test_predict_is_503_without_a_model(monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 503


def test_predict_never_forwards_post_booking_fields_to_the_model(monkeypatch):
    """Old clients may still send the leaky fields; they are ignored, not 422'd,
    and the model never sees them."""
    captured = {}

    class _Model:
        classes_ = ["Direct", "Online TA"]

        def predict_proba(self, frame):
            captured["columns"] = set(frame.columns)
            return [[0.2, 0.8]]

    monkeypatch.setattr(main, "model", _Model())
    legacy_payload = {
        **PAYLOAD,
        "distribution_channel": "Direct",
        "reservation_status": "Check-Out",
        "is_canceled": 0,
        "assigned_room_type": "C",
    }

    response = client.post("/predict", json=legacy_payload)

    assert response.status_code == 200
    assert response.json()["predicted_market_segment"] == "Online TA"
    assert captured["columns"].isdisjoint(
        {"distribution_channel", "reservation_status", "is_canceled", "assigned_room_type"}
    )


def test_predict_rejects_a_negative_adr():
    response = client.post("/predict", json={**PAYLOAD, "adr": -5.0})

    assert response.status_code == 422


def test_a_model_failure_is_a_500_without_internal_details(monkeypatch):
    broken = MagicMock()
    broken.classes_ = ["Direct"]
    broken.predict_proba.side_effect = KeyError("column 'secret_internal_name' missing")
    monkeypatch.setattr(main, "model", broken)

    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 500
    assert "secret_internal_name" not in response.text


def test_refresh_loop_does_not_redownload_an_unchanged_version(monkeypatch):
    """Re-reading the alias every interval must not mean re-downloading the model."""
    monkeypatch.setattr(main, "model", MagicMock())
    monkeypatch.setattr(main, "model_version", "3")
    monkeypatch.setattr(main, "MODEL_REFRESH_SECONDS", 0.01)
    monkeypatch.setattr(main, "_resolve_model_version", lambda: "3")
    loads: list[str] = []
    monkeypatch.setattr(main, "_load_model_with_retry", lambda uri: loads.append(uri))

    async def _run_briefly() -> None:
        task = asyncio.create_task(main._model_loader_loop())
        await asyncio.sleep(0.1)
        task.cancel()

    asyncio.run(_run_briefly())

    assert loads == []


def test_refresh_loop_loads_the_newly_promoted_version_by_number(monkeypatch):
    monkeypatch.setattr(main, "model", MagicMock())
    monkeypatch.setattr(main, "model_version", "3")
    monkeypatch.setattr(main, "MODEL_REFRESH_SECONDS", 0.01)
    monkeypatch.setattr(main, "_resolve_model_version", lambda: "4")
    new_model = MagicMock()
    loads: list[str] = []

    def _load(uri: str):
        loads.append(uri)
        return new_model

    monkeypatch.setattr(main, "_load_model_with_retry", _load)

    async def _run_briefly() -> None:
        task = asyncio.create_task(main._model_loader_loop())
        await asyncio.sleep(0.1)
        task.cancel()

    asyncio.run(_run_briefly())

    assert loads and loads[0] == f"models:/{main.MODEL_NAME}/4"
    assert main.model is new_model
    assert main.model_version == "4"
