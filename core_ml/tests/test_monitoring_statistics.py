"""Properties of the drift statistics, not their exact values.

A test that pins PSI to 0.1834 for a fixed input passes forever and catches
nothing, because the number it asserts is whatever the implementation happened
to produce on the day it was written. These assert the properties an operator
actually relies on when reading a report: identical distributions score zero, a
bigger shift scores higher, an unseen category is visible, and no input makes
the statistic infinite or NaN.
"""

import numpy as np
import pytest

from src.monitoring.profile import OTHER_CATEGORY
from src.monitoring.statistics import (
    PSI_ALERT_THRESHOLD,
    PSI_WARN_THRESHOLD,
    align_categorical,
    bootstrap_metric_ci,
    classify,
    jensen_shannon_distance,
    population_stability_index,
)


def test_psi_is_zero_for_identical_distributions():
    reference = [0.1, 0.2, 0.3, 0.25, 0.15]
    assert population_stability_index(reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_monotonic_in_the_size_of_the_shift():
    """The whole point of a threshold is that a bigger move scores higher."""
    reference = [0.25, 0.25, 0.25, 0.25]
    small_shift = [0.30, 0.25, 0.25, 0.20]
    large_shift = [0.70, 0.10, 0.10, 0.10]

    assert population_stability_index(reference, small_shift) < population_stability_index(
        reference, large_shift
    )


def test_psi_is_finite_when_a_bin_empties_completely():
    """An empty bin is log(0). Without smoothing this is +inf and the report is unreadable.

    Regression guard for the failure mode that makes hand-rolled PSI
    implementations useless in production: one rare category disappearing from a
    window is routine, and it must not swamp every other feature in the ranking.
    """
    reference = [0.25, 0.25, 0.25, 0.25]
    current = [0.5, 0.5, 0.0, 0.0]

    psi = population_stability_index(reference, current)

    assert np.isfinite(psi)
    assert psi > PSI_ALERT_THRESHOLD


def test_psi_handles_an_all_zero_current_window():
    """No rows in a bin set at all - must not divide by zero."""
    assert np.isfinite(population_stability_index([0.5, 0.5], [0.0, 0.0]))


def test_jensen_shannon_is_bounded_and_zero_for_identical_inputs():
    reference = [0.2, 0.3, 0.5]
    assert jensen_shannon_distance(reference, reference) == pytest.approx(0.0, abs=1e-9)

    disjoint = jensen_shannon_distance([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    assert 0.0 <= disjoint <= 1.0


def test_classify_matches_the_documented_bands():
    assert classify(PSI_WARN_THRESHOLD - 0.001) == "OK"
    assert classify(PSI_WARN_THRESHOLD) == "WARN"
    assert classify(PSI_ALERT_THRESHOLD) == "ALERT"


def test_align_categorical_folds_unseen_levels_into_other_and_reports_them():
    """A level the baseline never saw must be counted, not silently dropped.

    It is folded into the same bucket the baseline uses for its own tail so PSI
    scales with the *mass* the new levels carry rather than with how many
    distinct new levels appeared - otherwise a handful of one-off typos would
    outrank a genuine channel shift.
    """
    reference = {"Direct": 0.6, "Online TA": 0.4}
    current = ["Direct", "Online TA", "Aviation", "Aviation"]

    reference_counts, current_counts, unseen = align_categorical(reference, current)

    assert unseen == ["Aviation"]
    assert current_counts.sum() == 4
    other_index = -1
    assert current_counts[other_index] == 2
    assert len(reference_counts) == len(current_counts)


def test_align_categorical_keeps_the_baselines_own_other_bucket():
    reference = {"Direct": 0.5, OTHER_CATEGORY: 0.5}
    _, current_counts, unseen = align_categorical(reference, ["Direct", "Groups"])

    assert unseen == ["Groups"]
    assert current_counts[-1] == 1


def _accuracy(truth, predicted):
    return float((truth == predicted).mean())


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, size=500)
    predicted = truth.copy()
    predicted[:100] = (predicted[:100] + 1) % 3  # 80% accurate

    interval = bootstrap_metric_ci(truth, predicted, _accuracy, n_resamples=200)

    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.n == 500
    assert interval.estimate == pytest.approx(0.8, abs=0.01)


def test_bootstrap_interval_is_wider_on_a_smaller_sample():
    """Small samples must produce wide intervals - that is what stops the
    monitor from calling concept drift on forty labelled rows."""
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, size=1000)
    predicted = truth.copy()
    predicted[::5] = 1 - predicted[::5]

    wide = bootstrap_metric_ci(truth[:50], predicted[:50], _accuracy, n_resamples=300)
    narrow = bootstrap_metric_ci(truth, predicted, _accuracy, n_resamples=300)

    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_bootstrap_is_reproducible():
    """A monitoring verdict has to be the same on a re-run over the same window."""
    truth = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 20)
    predicted = np.array([0, 1, 1, 1, 0, 0, 1, 0] * 20)

    first = bootstrap_metric_ci(truth, predicted, _accuracy, n_resamples=100)
    second = bootstrap_metric_ci(truth, predicted, _accuracy, n_resamples=100)

    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_bootstrap_on_an_empty_sample_does_not_raise():
    interval = bootstrap_metric_ci(np.array([]), np.array([]), lambda a, b: 0.0, n_resamples=10)
    assert interval.n == 0
