from pathlib import Path

import pandas as pd
import pytest

from src.data_contracts import DataContractError
from src.data_processing import load_and_clean_data

BASE_ROW = {
    "hotel": "City Hotel",
    "market_segment": "Direct",
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "lead_time": 1,
    "adults": 2,
    "babies": 0,
    "children": 0.0,
    "adr": 100.0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "total_of_special_requests": 0,
}


def _rows(*overrides: dict) -> pd.DataFrame:
    return pd.DataFrame([{**BASE_ROW, **override} for override in overrides])


def test_load_and_clean_data_maps_months_and_drops_duplicates(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    df = _rows(
        {"lead_time": 1},
        {"lead_time": 2},
        {"lead_time": 3},
        {"lead_time": 3},  # exact duplicate of the row above
        {"market_segment": "Corporate", "arrival_date_month": "March", "lead_time": 5},
    )
    df.to_csv(raw_csv, index=False)

    # min_class_count=2 keeps "Direct" (3 unique rows after dedup) and drops
    # the rare "Corporate" class (only 1 occurrence).
    cleaned = load_and_clean_data(str(raw_csv), min_class_count=2)

    assert "market_segment" in cleaned.columns
    assert "month" in cleaned.columns
    assert "arrival_date_month" not in cleaned.columns
    assert set(cleaned["market_segment"].unique()) == {"Direct"}
    assert len(cleaned) == 3
    assert (cleaned["month"] == 7).all()


def test_load_and_clean_data_drops_negative_adr_rows(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    df = _rows(
        {"adr": 100.0},
        {"adr": 120.0},
        {"adr": -6.38},
    )
    df.to_csv(raw_csv, index=False)

    cleaned = load_and_clean_data(str(raw_csv), min_class_count=1)

    assert (cleaned["adr"] >= 0).all()
    assert len(cleaned) == 2


def test_load_and_clean_data_fails_fast_on_missing_required_column(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    df = _rows({}).drop(columns=["adr"])
    df.to_csv(raw_csv, index=False)

    with pytest.raises(DataContractError):
        load_and_clean_data(str(raw_csv), min_class_count=1)


def test_load_and_clean_data_fails_fast_on_out_of_range_value(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    df = _rows({"arrival_date_week_number": 99})  # only 1-53 is valid
    df.to_csv(raw_csv, index=False)

    with pytest.raises(DataContractError):
        load_and_clean_data(str(raw_csv), min_class_count=1)
