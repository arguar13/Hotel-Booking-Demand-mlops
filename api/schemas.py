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


class PredictionResponse(BaseModel):
    """What /predict answers.

    `prediction_id` is the contract that makes concept-drift monitoring
    possible at all: it is the key the caller quotes back on /feedback once the
    booking's true market segment is known, and the key the drift monitor joins
    on. Without it a prediction log can never be reconciled with ground truth.

    `confidence` and `margin` come from the classifier's `predict_proba` and are
    the standard unsupervised proxy for concept drift while labels are still in
    flight: a model whose decisions are getting less confident, or whose top two
    classes are converging, is usually a model whose world has moved. They are
    optional because a future estimator without `predict_proba` must degrade to
    a plain label rather than break the endpoint.
    """

    predicted_market_segment: str = Field(..., examples=["Direct"])
    prediction_id: str = Field(..., examples=["6f1c3f2a-6b1e-4a9e-9a8d-1f0f0f4a2b77"])
    model_version: str | None = Field(default=None, examples=["7"])
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, examples=[0.92])
    margin: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.84],
        description="Top-1 minus top-2 class probability. Near 0 means the model was "
        "effectively guessing between two segments, which a single confidence "
        "figure hides.",
    )


class LabelIngest(BaseModel):
    """One reconciled ground-truth segment for a previously served prediction."""

    prediction_id: str = Field(..., examples=["6f1c3f2a-6b1e-4a9e-9a8d-1f0f0f4a2b77"])
    actual_market_segment: str = Field(..., examples=["Online TA"])
    label_source: str = Field(
        default="reconciliation",
        examples=["reconciliation"],
        description="Where the truth came from - channel reconciliation, a PMS export, "
        "a manual correction. Recorded so a drift alert traced back to a single "
        "mislabelling source can be identified as such.",
    )


class LabelIngestResponse(BaseModel):
    """How much of the batch landed, and how much the caller should re-post.

    `skipped` is not an error. Predictions are persisted asynchronously, so a
    caller that reconciles quickly can quote an id that has not reached the
    table yet; those rows are skipped rather than rejected, and re-posting the
    batch is safe because the write is an idempotent upsert. A `skipped` count
    that stays non-zero across retries means those ids were never served.
    """

    accepted: int = Field(..., examples=[250])
    skipped: int = Field(
        default=0,
        examples=[0],
        description="Rows whose prediction_id was not in the prediction log yet. Re-post them.",
    )
