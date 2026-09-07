"""The decision logic - the part that decides whether to wake somebody up.

These are the tests that matter most in this package. The statistics can be
verified against a textbook; the escalation rules cannot, and getting them wrong
has two failure modes that are both worse than having no monitor:

  * escalating on noise, until the alert is routinely ignored;
  * staying quiet through a real regression because a guard was too coarse.

Each test below pins one of the rules that separates those two.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_monitor import (
    _worst,
    compute_concept_drift,
    compute_confidence_drift,
    compute_data_drift,
    compute_prediction_drift,
)
from src.monitoring.profile import PerformanceBaseline, build_reference_profile


@pytest.fixture
def profile():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "lead_time": rng.integers(0, 200, size=800),
            "adr": rng.normal(100, 20, size=800),
            "hotel": rng.choice(["Resort Hotel", "City Hotel"], size=800),
        }
    )
    target = pd.Series(rng.choice(["Direct", "Online TA", "Groups"], size=800))
    return build_reference_profile(
        frame,
        target,
        target_column="market_segment",
        performance=PerformanceBaseline(
            f1_weighted=0.90,
            accuracy=0.90,
            mean_confidence=0.80,
            mean_margin=0.50,
            n_eval=200,
        ),
    )


def _labelled(n: int, accuracy: float, seed: int = 11) -> pd.DataFrame:
    """A labelled window where a known share of predictions are correct."""
    rng = np.random.default_rng(seed)
    segments = np.array(["Direct", "Online TA", "Groups"])
    actual = rng.choice(segments, size=n)
    predicted = actual.copy()
    wrong = rng.random(n) > accuracy
    predicted[wrong] = rng.choice(segments, size=int(wrong.sum()))
    return pd.DataFrame({"actual_segment": actual, "predicted_segment": predicted})


# --- concept drift ---------------------------------------------------------


def test_concept_drift_is_skipped_below_the_label_minimum(profile):
    """Too few labels must read SKIPPED, never OK.

    'We could not tell' and 'we checked and it is fine' call for opposite
    responses, and collapsing them is how a monitor reports all-clear through an
    outage that stopped the labels arriving.
    """
    result = compute_concept_drift(
        profile,
        _labelled(50, 0.9),
        min_labelled=300,
        warn_tolerance=0.03,
        alert_tolerance=0.05,
        bootstrap_resamples=50,
    )

    assert result.status == "SKIPPED"
    assert result.live_f1 is None
    assert "waiting for reconciliation" in result.detail


def test_concept_drift_is_ok_when_live_performance_matches_the_baseline(profile):
    result = compute_concept_drift(
        profile,
        _labelled(1000, 0.90),
        min_labelled=300,
        warn_tolerance=0.03,
        alert_tolerance=0.05,
        bootstrap_resamples=200,
    )

    assert result.status == "OK"
    assert result.live_f1 == pytest.approx(0.90, abs=0.05)


def test_concept_drift_alerts_on_a_large_and_conclusive_drop(profile):
    """The finding this whole system exists to produce."""
    result = compute_concept_drift(
        profile,
        _labelled(2000, 0.55),
        min_labelled=300,
        warn_tolerance=0.03,
        alert_tolerance=0.05,
        bootstrap_resamples=200,
    )

    assert result.status == "ALERT"
    assert result.statistically_significant
    assert result.f1_drop > 0.05
    assert result.live_f1_upper < result.baseline_f1


def test_a_large_drop_on_a_thin_sample_is_only_a_warning(profile):
    """Material but not conclusive must not escalate.

    With a few hundred labels the bootstrap interval still reaches the baseline,
    so the honest response is to wait for more labels rather than pull a model
    that may be fine. Reporting this as an ALERT is how on-call learns to ignore
    ALERTs.
    """
    result = compute_concept_drift(
        profile,
        _labelled(320, 0.86),
        min_labelled=300,
        warn_tolerance=0.03,
        alert_tolerance=0.05,
        bootstrap_resamples=300,
    )

    assert result.status in ("OK", "WARN")
    assert result.status != "ALERT"


def test_concept_drift_is_skipped_when_the_model_has_no_baseline(profile):
    """A model version trained before baselining existed is a gap in coverage,
    not a crash - and the operator has to be told the fix is a retrain."""
    profile.performance = None

    result = compute_concept_drift(
        profile,
        _labelled(1000, 0.5),
        min_labelled=300,
        warn_tolerance=0.03,
        alert_tolerance=0.05,
        bootstrap_resamples=50,
    )

    assert result.status == "SKIPPED"
    assert "no performance baseline" in result.detail


# --- data drift ------------------------------------------------------------


def test_data_drift_is_ok_when_the_window_matches_the_baseline(profile):
    rng = np.random.default_rng(3)
    current = pd.DataFrame(
        {
            "lead_time": rng.integers(0, 200, size=800),
            "adr": rng.normal(100, 20, size=800),
            "hotel": rng.choice(["Resort Hotel", "City Hotel"], size=800),
        }
    )

    results = compute_data_drift(profile, current)

    assert {d.feature for d in results} == {"lead_time", "adr", "hotel"}
    assert all(d.status == "OK" for d in results)


def test_data_drift_flags_a_shifted_numeric_feature(profile):
    rng = np.random.default_rng(4)
    current = pd.DataFrame(
        {
            "lead_time": rng.integers(0, 200, size=800),
            "adr": rng.normal(220, 20, size=800),  # a different price regime
            "hotel": rng.choice(["Resort Hotel", "City Hotel"], size=800),
        }
    )

    by_feature = {d.feature: d for d in compute_data_drift(profile, current)}

    assert by_feature["adr"].status == "ALERT"
    assert by_feature["lead_time"].status == "OK"


def test_a_feature_that_stopped_arriving_is_an_alert_not_a_skip(profile):
    """A missing feature is a broken serving contract - the most urgent finding
    here, and the one a 'skip columns we do not have' shortcut would hide."""
    current = pd.DataFrame({"lead_time": np.arange(500) % 200})

    by_feature = {d.feature: d for d in compute_data_drift(profile, current)}

    assert by_feature["adr"].status == "ALERT"
    assert by_feature["adr"].n_current == 0
    assert "absent from the live window" in by_feature["adr"].detail


def test_unseen_categorical_levels_are_reported(profile):
    current = pd.DataFrame(
        {
            "lead_time": np.arange(400) % 200,
            "adr": np.full(400, 100.0),
            "hotel": ["Boutique Hotel"] * 400,
        }
    )

    by_feature = {d.feature: d for d in compute_data_drift(profile, current)}

    assert by_feature["hotel"].new_categories == ["Boutique Hotel"]
    assert by_feature["hotel"].status == "ALERT"


# --- prediction and confidence drift ---------------------------------------


def test_prediction_drift_detects_a_collapsed_output_distribution(profile):
    balanced = pd.Series(["Direct", "Online TA", "Groups"] * 300)
    collapsed = pd.Series(["Online TA"] * 900)

    stable_psi, stable_status = compute_prediction_drift(profile, balanced)
    shifted_psi, shifted_status = compute_prediction_drift(profile, collapsed)

    assert stable_status == "OK"
    assert shifted_status == "ALERT"
    assert shifted_psi > stable_psi


def test_confidence_drift_warns_only_past_the_relative_tolerance(profile):
    steady = pd.DataFrame({"confidence": [0.79] * 100, "margin": [0.49] * 100})
    collapsed = pd.DataFrame({"confidence": [0.55] * 100, "margin": [0.10] * 100})

    assert compute_confidence_drift(profile, steady, 0.10)["status"] == "OK"

    result = compute_confidence_drift(profile, collapsed, 0.10)
    assert result["status"] == "WARN"
    assert result["relative_drop"] > 0.10


def test_confidence_drift_is_skipped_without_a_baseline(profile):
    profile.performance.mean_confidence = None
    frame = pd.DataFrame({"confidence": [0.5] * 10, "margin": [0.1] * 10})

    assert compute_confidence_drift(profile, frame, 0.10)["status"] == "SKIPPED"


# --- severity ordering -----------------------------------------------------


def test_worst_status_ordering():
    assert _worst(["OK", "WARN", "SKIPPED"]) == "WARN"
    assert _worst(["OK", "ALERT", "WARN"]) == "ALERT"
    assert _worst(["SKIPPED"]) == "SKIPPED"
    assert _worst([]) == "SKIPPED"


# --- the artifact an operator actually opens --------------------------------


def _report(**overrides):
    from src.monitoring.drift_monitor import ConceptDrift, DriftReport

    defaults: dict = {
        "status": "ALERT",
        "reason": "concept drift (ALERT): weighted F1 fell below baseline",
        "model_name": "HotelSegmentClassifier",
        "model_alias": "staging",
        "model_version": "7",
        "window_start": "2026-08-29T04:00:00+00:00",
        "window_end": "2026-08-30T04:00:00+00:00",
        "coverage": {"n_predictions": 4200, "n_labelled": 3100, "label_coverage": 0.738},
        "data_drift": [],
        "drift_share": 0.33,
        "prediction_drift_psi": 0.31,
        "prediction_drift_status": "ALERT",
        "confidence_drift": {"status": "WARN", "reason": "mean confidence fell 12%"},
        "concept_drift": ConceptDrift(
            status="ALERT",
            n_labelled=3100,
            baseline_f1=0.90,
            live_f1=0.81,
            live_f1_lower=0.79,
            live_f1_upper=0.83,
            f1_drop=0.09,
            live_accuracy=0.82,
            statistically_significant=True,
            per_class_f1={"Direct": 0.7, "Online TA": 0.9},
            detail="the model's relationship to its target has changed",
        ),
        "consecutive_alerts": 1,
        "generated_at": "2026-08-30T04:01:00+00:00",
    }
    defaults.update(overrides)
    return DriftReport(**defaults)


def test_the_html_report_renders_a_full_verdict():
    """It is written at the very end of the job, after all the analysis - a
    rendering bug here throws away a run that had already done its work."""
    from src.monitoring.report import render_html_report

    html = render_html_report(_report())

    assert "<!doctype html>" in html
    assert "HotelSegmentClassifier" in html
    assert "ALERT" in html
    assert "0.9000" in html  # the baseline F1
    assert "Online TA" in html


def test_the_html_report_renders_a_skipped_run_with_no_analysis():
    """The most common outcome on a quiet day must not need a special path."""
    from src.monitoring.drift_monitor import ConceptDrift
    from src.monitoring.report import render_html_report

    html = render_html_report(
        _report(
            status="SKIPPED",
            reason="12 prediction(s) in the window, 500 required",
            prediction_drift_psi=None,
            prediction_drift_status="SKIPPED",
            confidence_drift={"status": "SKIPPED", "reason": "no traffic"},
            concept_drift=ConceptDrift(
                status="SKIPPED",
                n_labelled=0,
                baseline_f1=None,
                live_f1=None,
                live_f1_lower=None,
                live_f1_upper=None,
                f1_drop=None,
                live_accuracy=None,
                statistically_significant=False,
                detail="waiting for reconciliation",
            ),
        )
    )

    assert "SKIPPED" in html
    assert "n/a" in html


def test_the_report_serialises_to_json_for_machine_consumption():
    """The JSON artifact is what a future alerting integration would read."""
    import json

    payload = _report().to_dict()

    assert json.loads(json.dumps(payload, default=str))["concept_drift"]["status"] == "ALERT"


def test_a_change_in_missingness_is_reported_as_its_own_finding():
    """Found by the first end-to-end run, not by a unit test.

    Training saw `company` null for ~94% of rows; a caller that substitutes 0.0
    for an absent value sends it null for 0%. Both sides drop nulls before
    binning, so this produced a PSI of 5.6 whose size was an artefact of
    comparing two different populations rather than a measure of how far
    anything moved. Naming the missingness change is what turns that number into
    the actual finding - the serving contract and the training data disagree.
    """
    reference = pd.DataFrame({"company": [np.nan] * 940 + list(range(60))})
    profile_with_gaps = build_reference_profile(
        reference,
        pd.Series(["Direct"] * 1000),
        target_column="market_segment",
    )
    current = pd.DataFrame({"company": [0.0] * 800})

    result = compute_data_drift(profile_with_gaps, current)[0]

    assert result.status == "ALERT"
    assert "train/serve skew" in result.detail
    assert "94% -> 0%" in result.detail
