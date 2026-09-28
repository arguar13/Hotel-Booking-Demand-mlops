"""Data contracts for the hotel booking pipeline.

Every DataFrame that enters an expensive stage (Optuna tuning, model
fit) must pass one of these schemas first. Validation is deliberately cheap
(column presence, dtype, range, nullability) so that a malformed dataset is
rejected in milliseconds instead of after minutes of training - fail fast.
"""

import logging

import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)


class DataContractError(Exception):
    """Raised when a DataFrame violates its data contract."""


# Schema for the data as it comes out of the raw CSV, before any cleaning.
# Deliberately loose: this stage only guarantees the columns the cleaning
# step depends on actually exist and hold plausible values.
RawBookingSchema = pa.DataFrameSchema(
    {
        "hotel": pa.Column(str, nullable=False),
        "market_segment": pa.Column(str, nullable=False),
        "arrival_date_month": pa.Column(str, nullable=False),
        "lead_time": pa.Column(int, pa.Check.ge(0)),
        "adults": pa.Column(int, pa.Check.ge(0)),
        "babies": pa.Column(int, pa.Check.ge(0)),
        "children": pa.Column(float, pa.Check.ge(0), nullable=True),
        "adr": pa.Column(float, nullable=True),  # validated post-cleaning, see below
        "arrival_date_week_number": pa.Column(int, pa.Check.in_range(1, 53)),
        "arrival_date_day_of_month": pa.Column(int, pa.Check.in_range(1, 31)),
    },
    strict=False,  # the raw file has ~30 columns; we only pin down the ones we use
    coerce=False,
)

# Schema for the cleaned/processed dataset, right before it is split and fed
# into the training pipeline. Stricter: by this point every row must be
# usable, because the next step (Optuna + RandomForest) is expensive.
#
# Covers every model input column declared in config.yaml's `features:`
# block, so a dataset that is missing one fails here rather than as a
# KeyError deep inside the fitted pipeline.
ProcessedBookingSchema = pa.DataFrameSchema(
    {
        "market_segment": pa.Column(str, nullable=False),
        # categorical inputs (country has genuine gaps in the source data)
        "hotel": pa.Column(str, nullable=False),
        "meal": pa.Column(str, nullable=False),
        "country": pa.Column(str, nullable=True),
        "reserved_room_type": pa.Column(str, nullable=False),
        "deposit_type": pa.Column(str, nullable=False),
        "customer_type": pa.Column(str, nullable=False),
        # cyclical inputs
        "month": pa.Column(int, pa.Check.in_range(1, 12)),
        "arrival_date_week_number": pa.Column(int, pa.Check.in_range(1, 53)),
        # numeric inputs
        "arrival_date_day_of_month": pa.Column(int, pa.Check.in_range(1, 31)),
        "lead_time": pa.Column(int, pa.Check.ge(0)),
        "stays_in_weekend_nights": pa.Column(int, pa.Check.ge(0)),
        "stays_in_week_nights": pa.Column(int, pa.Check.ge(0)),
        "adults": pa.Column(int, pa.Check.ge(0)),
        "children": pa.Column(float, pa.Check.ge(0), nullable=True),
        "babies": pa.Column(int, pa.Check.ge(0)),
        "is_repeated_guest": pa.Column(int, pa.Check.isin([0, 1])),
        "previous_cancellations": pa.Column(int, pa.Check.ge(0)),
        "previous_bookings_not_canceled": pa.Column(int, pa.Check.ge(0)),
        "days_in_waiting_list": pa.Column(int, pa.Check.ge(0)),
        "adr": pa.Column(float, pa.Check.ge(0)),
        "required_car_parking_spaces": pa.Column(int, pa.Check.ge(0)),
        "total_of_special_requests": pa.Column(int, pa.Check.ge(0)),
        # identifier columns: NaN means "no agent / no company", so nullable
        "agent": pa.Column(float, pa.Check.ge(1), nullable=True),
        "company": pa.Column(float, pa.Check.ge(1), nullable=True),
    },
    strict=False,
    coerce=False,
    checks=[
        pa.Check(lambda df: len(df) > 0, error="processed dataset must not be empty"),
    ],
)


def _validate(df: pd.DataFrame, schema: pa.DataFrameSchema, stage: str) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as err:
        logger.error("Data contract violated at stage '%s':\n%s", stage, err.failure_cases)
        raise DataContractError(
            f"Data contract violated at stage '{stage}': {len(err.failure_cases)} failing case(s). "
            f"See logs for details."
        ) from err


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fail fast on the raw CSV, before spending time cleaning it."""
    return _validate(df, RawBookingSchema, stage="raw")


def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Fail fast on the cleaned dataset, before spending compute training on it."""
    return _validate(df, ProcessedBookingSchema, stage="processed")
