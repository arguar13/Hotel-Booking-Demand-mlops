"""drift_check.py: build a reference profile, then check a batch against it.

These tests only exercise the pure functions (profile building, the z-test,
JSON round-trip) - no database, no MLflow server needed.

The reference profile is built from a few thousand rows, not a couple
hundred: the z-test's standard error uses the reference's own std, so a
reference built from too few rows carries enough sampling noise in its own
mean estimate to make an otherwise-matching batch look drifted. A few
thousand rows (still a fraction of a real training set) is enough for the
statistic to behave the way the docstring in drift_check.py describes.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_check import (
    PerformanceBaseline,
    ReferenceProfile,
    build_reference_profile,
    check_batch,
)

N_REFERENCE_ROWS = 5000


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "lead_time": rng.normal(100, 20, size=N_REFERENCE_ROWS),
            "adr": rng.normal(100, 30, size=N_REFERENCE_ROWS),
            "hotel": rng.choice(["Resort Hotel", "City Hotel"], size=N_REFERENCE_ROWS),
        }
    )


def test_build_profile_only_covers_numeric_columns(frame):
    profile = build_reference_profile(frame)

    assert set(profile.numeric) == {"lead_time", "adr"}
    assert profile.n_rows == N_REFERENCE_ROWS


def test_profile_round_trips_through_json(frame, tmp_path):
    original = build_reference_profile(
        frame,
        performance=PerformanceBaseline(f1_weighted=0.87, accuracy=0.88, n_eval=100),
    )
    path = original.write(tmp_path / "reference_profile.json")
    restored = ReferenceProfile.read(path)

    assert restored.numeric.keys() == original.numeric.keys()
    assert restored.performance is not None
    assert restored.performance.f1_weighted == pytest.approx(0.87)
    assert restored.numeric["adr"].mean == pytest.approx(original.numeric["adr"].mean)


def test_profile_without_performance_round_trips(frame, tmp_path):
    profile = build_reference_profile(frame)
    restored = ReferenceProfile.read(profile.write(tmp_path / "p.json"))

    assert restored.performance is None


def test_check_batch_is_ok_when_the_batch_matches_the_reference(frame):
    profile = build_reference_profile(frame)
    rng = np.random.default_rng(101)
    same_distribution = pd.DataFrame(
        {
            "lead_time": rng.normal(100, 20, size=500),
            "adr": rng.normal(100, 30, size=500),
        }
    )

    result = check_batch(profile, same_distribution)

    assert result.status == "OK"
    assert all(f.status == "OK" for f in result.features)


def test_check_batch_flags_a_shifted_feature(frame):
    profile = build_reference_profile(frame)
    rng = np.random.default_rng(1)
    shifted = pd.DataFrame(
        {
            "lead_time": rng.normal(100, 20, size=500),
            "adr": rng.normal(250, 30, size=500),  # a very different price regime
        }
    )

    result = check_batch(profile, shifted)
    by_feature = {f.feature: f for f in result.features}

    assert result.status == "ALERT"
    assert by_feature["adr"].status == "ALERT"
    assert by_feature["lead_time"].status == "OK"


def test_check_batch_is_skipped_when_no_feature_overlaps(frame):
    profile = build_reference_profile(frame)
    batch = pd.DataFrame({"unrelated_column": [1, 2, 3]})

    result = check_batch(profile, batch)

    assert result.status == "SKIPPED"


def test_check_batch_ignores_features_with_too_few_values(frame):
    profile = build_reference_profile(frame)
    batch = pd.DataFrame({"adr": [100.0]})  # a single row can't support a z-test

    result = check_batch(profile, batch)

    assert result.status == "SKIPPED"
