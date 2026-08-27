from pydantic import BaseModel, Field


class BookingFeatures(BaseModel):
    """
    Pydantic schema for validating hotel booking input data based on expected model features.

    Pydantic v2: field examples go in `examples=[...]` (a list), not the v1
    `example=` kwarg - FastAPI renders these into the OpenAPI schema shown at
    /docs.
    """

    hotel: str = Field(..., examples=["Resort Hotel"])
    lead_time: int = Field(..., ge=0, examples=[342])
    arrival_date_year: int = Field(..., examples=[2015])
    arrival_date_week_number: int = Field(..., ge=1, le=53, examples=[27])
    arrival_date_day_of_month: int = Field(..., ge=1, le=31, examples=[1])
    stays_in_weekend_nights: int = Field(..., ge=0, examples=[0])
    stays_in_week_nights: int = Field(..., ge=0, examples=[0])
    adults: int = Field(..., ge=0, examples=[2])
    children: float = Field(default=0.0, ge=0.0, examples=[0.0])
    babies: int = Field(default=0, ge=0, examples=[0])
    meal: str = Field(..., examples=["BB"])
    country: str = Field(..., examples=["PRT"])
    distribution_channel: str = Field(..., examples=["Direct"])
    is_repeated_guest: int = Field(default=0, examples=[0])
    previous_cancellations: int = Field(default=0, examples=[0])
    previous_bookings_not_canceled: int = Field(default=0, examples=[0])
    reserved_room_type: str = Field(..., examples=["C"])
    assigned_room_type: str = Field(..., examples=["C"])
    booking_changes: int = Field(default=0, examples=[3])
    deposit_type: str = Field(..., examples=["No Deposit"])
    days_in_waiting_list: int = Field(default=0, examples=[0])
    customer_type: str = Field(..., examples=["Transient"])
    adr: float = Field(..., examples=[0.0])
    required_car_parking_spaces: int = Field(default=0, examples=[0])
    total_of_special_requests: int = Field(default=0, examples=[0])
    month: int = Field(..., ge=1, le=12, examples=[7])
    is_canceled: int = Field(default=0, examples=[0])
    agent: float = Field(default=0.0, examples=[9.0])
    company: float = Field(default=0.0, examples=[0.0])
    # Optional variables
    reservation_status: str | None = None
    reservation_status_date: str | None = None
