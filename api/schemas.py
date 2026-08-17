from pydantic import BaseModel, Field
from typing import Optional

class BookingFeatures(BaseModel):
    """
    Pydantic schema for validating hotel booking input data based on expected model features.
    """
    hotel: str = Field(..., example="Resort Hotel")
    lead_time: int = Field(..., ge=0, example=342)
    arrival_date_year: int = Field(..., example=2015)
    arrival_date_week_number: int = Field(..., ge=1, le=53, example=27)
    arrival_date_day_of_month: int = Field(..., ge=1, le=31, example=1)
    stays_in_weekend_nights: int = Field(..., ge=0, example=0)
    stays_in_week_nights: int = Field(..., ge=0, example=0)
    adults: int = Field(..., ge=0, example=2)
    children: float = Field(default=0.0, ge=0.0, example=0.0)
    babies: int = Field(default=0, ge=0, example=0)
    meal: str = Field(..., example="BB")
    country: str = Field(..., example="PRT")
    distribution_channel: str = Field(..., example="Direct")
    is_repeated_guest: int = Field(default=0, example=0)
    previous_cancellations: int = Field(default=0, example=0)
    previous_bookings_not_canceled: int = Field(default=0, example=0)
    reserved_room_type: str = Field(..., example="C")
    assigned_room_type: str = Field(..., example="C")
    booking_changes: int = Field(default=0, example=3)
    deposit_type: str = Field(..., example="No Deposit")
    days_in_waiting_list: int = Field(default=0, example=0)
    customer_type: str = Field(..., example="Transient")
    adr: float = Field(..., example=0.0)
    required_car_parking_spaces: int = Field(default=0, example=0)
    total_of_special_requests: int = Field(default=0, example=0)
    month: int = Field(..., ge=1, le=12, example=7)
    is_canceled: int = Field(default=0, example=0)
    agent: float = Field(default=0.0, example=9.0)
    company: float = Field(default=0.0, example=0.0)
    # Optional variables
    reservation_status: Optional[str] = None
    reservation_status_date: Optional[str] = None