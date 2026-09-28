"""The feature contract (config.yaml `features:`) and the pipeline built from it."""

import pickle
import subprocess  # nosec B404 - runs this same interpreter with a fixed script
import sys

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from src import features as features_module
from src.config_loader import load_config
from src.features import BookingFeatureBuilder, FeatureSpec, build_model_pipeline

# Columns that are either only known after the booking exists or are
# near-copies of the target. See config.yaml for the reasoning per column.
LEAKY_COLUMNS = {
    "market_segment",
    "distribution_channel",
    "reservation_status",
    "reservation_status_date",
    "is_canceled",
    "assigned_room_type",
    "booking_changes",
    "arrival_date_year",
}


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return FeatureSpec.from_config(load_config())


def test_no_leaky_column_is_a_model_input(spec: FeatureSpec) -> None:
    assert LEAKY_COLUMNS.isdisjoint(spec.input_columns)


def test_identifier_columns_are_only_used_as_presence_flags(spec: FeatureSpec) -> None:
    for identifier in ("agent", "company"):
        assert identifier in spec.presence_flags
        assert identifier not in spec.numeric
        assert identifier not in spec.monitored_numeric_columns


def test_cyclical_encoding_puts_the_year_boundary_next_to_itself() -> None:
    frame = pd.DataFrame({"month": [1, 6, 12], "arrival_date_week_number": [1, 27, 53]})
    out = BookingFeatureBuilder(cyclical={"month": 12, "arrival_date_week_number": 53}).transform(
        frame
    )

    def distance(i: int, j: int, column: str) -> float:
        a = out.loc[i, [f"{column}_sin", f"{column}_cos"]].to_numpy(dtype=float)
        b = out.loc[j, [f"{column}_sin", f"{column}_cos"]].to_numpy(dtype=float)
        return float(np.linalg.norm(a - b))

    # December is as close to January as any two consecutive months, and much
    # closer than June - the property a raw 1..12 integer does not have.
    assert distance(0, 2, "month") < distance(0, 1, "month")
    assert distance(0, 2, "arrival_date_week_number") < distance(0, 1, "arrival_date_week_number")


def test_presence_flags_treat_nan_and_zero_as_absent() -> None:
    frame = pd.DataFrame({"agent": [9.0, np.nan, 0.0, None], "company": [np.nan, 40.0, np.nan, 1]})
    out = BookingFeatureBuilder(presence_flags=["agent", "company"]).transform(frame)

    assert out["has_agent"].tolist() == [1, 0, 0, 0]
    assert out["has_company"].tolist() == [0, 1, 0, 1]


def _bookings(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "lead_time": rng.integers(0, 300, n),
            "arrival_date_day_of_month": rng.integers(1, 29, n),
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(0, 6, n),
            "adults": rng.integers(1, 4, n),
            "children": rng.choice([0.0, 1.0, np.nan], n),
            "babies": 0,
            "is_repeated_guest": rng.integers(0, 2, n),
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "days_in_waiting_list": 0,
            "adr": rng.uniform(40, 200, n),
            "required_car_parking_spaces": 0,
            "total_of_special_requests": rng.integers(0, 3, n),
            "month": rng.integers(1, 13, n),
            "arrival_date_week_number": rng.integers(1, 54, n),
            "agent": rng.choice([9.0, 240.0, np.nan], n),
            "company": rng.choice([40.0, np.nan], n),
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "meal": rng.choice(["BB", "HB"], n),
            "country": rng.choice(["PRT", "GBR", None], n),
            "reserved_room_type": rng.choice(["A", "D"], n),
            "deposit_type": "No Deposit",
            "customer_type": rng.choice(["Transient", "Group"], n),
        }
    )
    target = pd.Series(rng.choice(["Online TA", "Direct", "Groups"], n))
    return frame, target


def test_pipeline_fits_and_tolerates_unseen_categories(spec: FeatureSpec) -> None:
    X, y = _bookings()
    pipeline = build_model_pipeline(spec, {"n_estimators": 10, "max_depth": 4}, random_state=0)
    pipeline.fit(X[spec.input_columns], y)

    unseen = X.head(2).copy()
    unseen["country"] = "ATA"  # never seen in training
    unseen["agent"] = None
    probabilities = pipeline.predict_proba(unseen[spec.input_columns])

    assert probabilities.shape == (2, 3)


def test_pipeline_ignores_extra_columns(spec: FeatureSpec) -> None:
    """The API logs and forwards whatever the schema accepts; anything outside
    the feature list must not change the prediction."""
    X, y = _bookings()
    pipeline = build_model_pipeline(spec, {"n_estimators": 10, "max_depth": 4}, random_state=0)
    pipeline.fit(X[spec.input_columns], y)

    with_extras = X.assign(distribution_channel="Direct", is_canceled=1)

    np.testing.assert_array_equal(pipeline.predict(X), pipeline.predict(with_extras))


def test_pipeline_loads_where_src_is_not_importable(spec: FeatureSpec, tmp_path) -> None:
    """The API image has no `src` package, so the artifact must embed the class.

    Loaded in a fresh interpreter with `src` blocked, which is exactly the
    situation inside the API container.
    """
    X, y = _bookings()
    pipeline = build_model_pipeline(spec, {"n_estimators": 5, "max_depth": 3}, random_state=0)
    pipeline.fit(X[spec.input_columns], y)

    cloudpickle.register_pickle_by_value(features_module)
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(cloudpickle.dumps(pipeline))
    sample = tmp_path / "sample.pkl"
    sample.write_bytes(pickle.dumps(X.head(5)))

    script = (
        "import sys, pickle; sys.modules['src'] = None\n"
        f"model = pickle.loads(open({str(artifact)!r}, 'rb').read())\n"
        f"frame = pickle.loads(open({str(sample)!r}, 'rb').read())\n"
        "print(','.join(model.predict(frame)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split(",") == list(pipeline.predict(X.head(5)))
