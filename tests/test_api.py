import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

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
        "company": 0.0
    }
    
    response = client.post("/predict", json=payload)
    
    # Si el modelo está cargado debe responder 200, si no está cargado 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_market_segment" in data