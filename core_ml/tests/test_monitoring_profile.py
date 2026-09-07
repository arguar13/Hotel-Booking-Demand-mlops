"""The reference baseline: it must survive a JSON round-trip and spiky data.

Both properties are load-bearing. The profile is written by the training job and
read back, days later, by a different container from an S3 artifact - so a field
that does not serialise is a CronJob crash at 04:00, not a unit-test failure.
And the features here are genuinely spiky (`babies` is zero for ~99% of rows),
which is exactly the input that breaks naive quantile binning.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.profile import (
    MAX_CATEGORY_LEVELS,
    OTHER_CATEGORY,
    PerformanceBaseline,
    ReferenceProfile,
    build_reference_profile,
    histogram,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 400, size=500),
            "adr": rng.normal(100, 30, size=500),
            # Spiky on purpose: >90% identical values, which produces duplicate
            # quantiles and would make np.histogram reject the edge array.
            "babies": np.where(rng.random(500) < 0.95, 0, 1),
            "hotel": rng.choice(["Resort Hotel", "City Hotel"], size=500),
            "country": rng.choice(["PRT", "GBR", "ESP", "FRA"], size=500),
        }
    )


@pytest.fixture
def target() -> pd.Series:
    rng = np.random.default_rng(8)
    return pd.Series(rng.choice(["Direct", "Online TA", "Groups"], size=500))


def test_build_profile_separates_numeric_and_categorical(frame, target):
    profile = build_reference_profile(frame, target, target_column="market_segment")

    assert set(profile.numeric) == {"lead_time", "adr", "babies"}
    assert set(profile.categorical) == {"hotel", "country"}
    assert profile.n_rows == 500
    assert sum(profile.target_frequencies.values()) == pytest.approx(1.0)


def test_quantile_edges_survive_a_spiky_feature(frame, target):
    """A feature where one value holds most of the mass produces duplicate
    quantiles; collapsing them is what keeps np.histogram from raising."""
    profile = build_reference_profile(frame, target, target_column="market_segment")
    edges = profile.numeric["babies"].bin_edges

    assert len(edges) >= 2
    assert all(later > earlier for earlier, later in zip(edges, edges[1:], strict=False))
    assert sum(profile.numeric["babies"].frequencies) == pytest.approx(1.0)


def test_outer_bin_edges_are_infinite_so_unseen_extremes_still_land_somewhere(frame, target):
    """An out-of-range live value is drift worth catching, not a row to drop."""
    profile = build_reference_profile(frame, target, target_column="market_segment")
    edges = profile.numeric["lead_time"].bin_edges

    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")

    frequencies = histogram(np.array([-9999.0, 9999.0]), edges)
    assert sum(frequencies) == pytest.approx(1.0)


def test_constant_feature_does_not_break_binning(target):
    frame = pd.DataFrame({"always_zero": np.zeros(100)})
    profile = build_reference_profile(frame, target.head(100), target_column="market_segment")

    assert "always_zero" in profile.numeric
    assert sum(profile.numeric["always_zero"].frequencies) == pytest.approx(1.0)


def test_high_cardinality_categorical_is_truncated_to_a_bounded_head(target):
    frame = pd.DataFrame({"agent": [f"agent-{i}" for i in range(500)]})
    profile = build_reference_profile(frame, target, target_column="market_segment")

    levels = profile.categorical["agent"].frequencies
    assert len(levels) == MAX_CATEGORY_LEVELS + 1
    assert OTHER_CATEGORY in levels
    assert sum(levels.values()) == pytest.approx(1.0)


def test_profile_round_trips_through_json(frame, target, tmp_path):
    """Written by the training container, read by the monitoring container."""
    original = build_reference_profile(
        frame,
        target,
        target_column="market_segment",
        performance=PerformanceBaseline(
            f1_weighted=0.87,
            accuracy=0.88,
            mean_confidence=0.71,
            mean_margin=0.42,
            n_eval=100,
        ),
    )
    path = original.write(tmp_path / "reference_profile.json")
    restored = ReferenceProfile.read(path)

    assert restored.feature_names == original.feature_names
    assert restored.performance is not None
    assert restored.performance.f1_weighted == pytest.approx(0.87)
    assert restored.performance.mean_margin == pytest.approx(0.42)
    assert restored.numeric["lead_time"].bin_edges == original.numeric["lead_time"].bin_edges
    # Infinities are the part most likely to be lost in serialisation, and
    # losing them would silently start dropping out-of-range live values.
    assert restored.numeric["lead_time"].bin_edges[0] == float("-inf")


def test_profile_without_performance_round_trips(frame, target, tmp_path):
    """A model trained before baselining had a performance section still loads."""
    profile = build_reference_profile(frame, target, target_column="market_segment")
    restored = ReferenceProfile.read(profile.write(tmp_path / "p.json"))

    assert restored.performance is None
