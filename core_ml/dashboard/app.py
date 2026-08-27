import os

import requests
import streamlit as st

st.set_page_config(page_title="MLOps Monitoring Dashboard", layout="wide")

st.title("Hotel Segmentation - MLOps Dashboard")

st.header("Test Model Inference")
st.write("Provide input data to test the FastAPI endpoint.")

# Sample inputs for testing
lead_time = st.number_input("Lead Time", min_value=0, value=10)
adults = st.number_input("Adults", min_value=0, value=2)
meal = st.selectbox("Meal", ["BB", "HB", "FB"])

if st.button("Predict"):
    # Payload matching the Pydantic schema
    payload = {
        "hotel": "Resort Hotel",
        "lead_time": lead_time,
        "arrival_date_year": 2015,
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2,
        "adults": adults,
        "children": 0,
        "babies": 0,
        "meal": meal,
        "country": "PRT",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 98.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "month": 7,
    }

    # Leer la URL desde la variable de entorno, con fallback para local
    api_url = os.getenv("API_URL", "http://localhost:8000")

    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            st.success(f"Predicted Market Segment: {response.json()['predicted_market_segment']}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
