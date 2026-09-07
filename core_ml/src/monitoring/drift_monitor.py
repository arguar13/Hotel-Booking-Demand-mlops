"""The drift monitor: one windowed evaluation of the model currently in production.

Runs as a Kubernetes CronJob (`kubernetes/base/drift-monitor-cronjob.yaml`),
answers four questions about one time window, records the answers as an MLflow
run, and exits.

    1. data drift        has P(X) moved away from what the model was fitted on?
    2. prediction drift  has P(y-hat) moved?
    3. confidence drift  is the model less sure than it was?
    4. CONCEPT drift     has P(y|X) moved - i.e. is the model still *right*?

Only (4) is concept drift in the strict sense, and only (4) requires ground
truth. The other three are computable from the serving side alone and act as
early warning during the reconciliation delay, before enough labels have
arrived to say anything conclusive. Conflating them is the single most common
way "drift monitoring" ends up reporting on inputs while a model quietly gets
worse, so they are reported and thresholded separately here, and the overall
verdict names which one fired.

**The verdict is deliberately hard to trigger.** Three mechanisms, each closing
a specific way monitors become noise and get ignored:

  * a minimum-sample guard - too little data yields SKIPPED, never "no drift";
  * effect sizes with fixed bands rather than p-values, so a large window
    cannot manufacture significance out of a difference nobody would act on;
  * hysteresis - a single ALERT window is reported but does not escalate;
    `consecutive_alerts_required` windows in a row must agree first.

**It never retrains or promotes anything.** The job's authority ends at
recording a finding and failing loudly. Closing the loop automatically -
drift fires, model retrains, alias moves - would let a data-quality incident
upstream promote a model trained on the incident, with no human between the
two. Retraining is a decision; this is the evidence for it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

from src.config_loader import load_config
from src.monitoring.profile import ReferenceProfile, histogram
from src.monitoring.report import render_html_report
from src.monitoring.statistics import (
    FeatureDrift,
    PerformanceInterval,
    align_categorical,
    bootstrap_metric_ci,
    classify,
    jensen_shannon_distance,
    population_stability_index,
)
from src.monitoring.store import MonitoringStore, Window
from src.traceability import get_git_commit_hash

log = structlog.get_logger("hotel_mlops.monitoring")

REFERENCE_ARTIFACT_PATH = "monitoring/reference_profile.json"

# Ordered worst-last so `max(..., key=SEVERITY.index)` gives the overall status.
SEVERITY = ["SKIPPED", "OK", "WARN", "ALERT"]

# How far a feature's missing rate may move before that is a finding in itself.
# Absolute percentage points, not relative: 2% missing becoming 12% is a far
# bigger operational change than 90% becoming 100%, and a relative measure would
# say the opposite.
MISSING_RATE_WARN = 0.10
MISSING_RATE_ALERT = 0.25


def _worst(statuses: list[str]) -> str:
    present = [s for s in statuses if s in SEVERITY]
    if not present:
        return "SKIPPED"
    return max(present, key=SEVERITY.index)


class InsufficientData(Exception):
    """Not enough rows in the window to say anything. Not a failure."""


@dataclass
class ConceptDrift:
    """The supervised verdict: is the model still right?"""

    status: str
    n_labelled: int
    baseline_f1: float | None
    live_f1: float | None
    live_f1_lower: float | None
    live_f1_upper: float | None
    f1_drop: float | None
    live_accuracy: float | None
    statistically_significant: bool
    per_class_f1: dict[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass
class DriftReport:
    status: str
    reason: str
    model_name: str
    model_alias: str
    model_version: str | None
    window_start: str
    window_end: str
    coverage: dict[str, Any]
    data_drift: list[FeatureDrift]
    drift_share: float
    prediction_drift_psi: float | None
    prediction_drift_status: str
    confidence_drift: dict[str, Any]
    concept_drift: ConceptDrift
    consecutive_alerts: int
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_drift"] = [asdict(d) for d in self.data_drift]
        payload["concept_drift"] = asdict(self.concept_drift)
        return payload


# ---------------------------------------------------------------------------
# baseline resolution
# ---------------------------------------------------------------------------


def resolve_served_version(client: MlflowClient, name: str, alias: str):
    """The model version the API is actually serving right now.

    The monitor deliberately measures what production serves, not the newest
    version in the registry: a version that was trained but never aliased has
    served no traffic, so there is nothing about it to monitor.
    """
    return client.get_model_version_by_alias(name, alias)


def load_reference_profile(run_id: str) -> ReferenceProfile:
    """Fetch the baseline that was logged alongside this exact model version."""
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=REFERENCE_ARTIFACT_PATH
    )
    return ReferenceProfile.read(local_path)


# ---------------------------------------------------------------------------
# the four analyses
# ---------------------------------------------------------------------------


def compute_data_drift(profile: ReferenceProfile, current: pd.DataFrame) -> list[FeatureDrift]:
    """Per-feature PSI of the live window against the training baseline.

    Features present in the baseline but absent from the live window are
    reported explicitly rather than skipped: a feature that stopped arriving is
    a more urgent finding than one whose distribution moved, and silently
    dropping it from the loop is how that goes unnoticed.
    """
    results: list[FeatureDrift] = []

    for name, baseline in profile.numeric.items():
        if name not in current.columns:
            results.append(_missing_feature(name, "numeric"))
            continue
        column = pd.to_numeric(current[name], errors="coerce")
        values = column.dropna().to_numpy(dtype=float)
        if values.size == 0:
            results.append(_missing_feature(name, "numeric"))
            continue
        current_frequencies = histogram(values, baseline.bin_edges)
        psi = population_stability_index(baseline.frequencies, current_frequencies)

        # Missingness is its own kind of drift, and reporting it separately is
        # what keeps the PSI above honest. Both sides drop nulls before binning,
        # so a feature that was 94% missing in training and is 0% missing live -
        # a caller substituting a default for an absent value - produces a huge
        # PSI whose magnitude is an artefact of comparing two different
        # populations, not a measure of how far the distribution moved. Naming
        # the missingness change turns that from a mysterious number into the
        # actual finding: the serving contract and the training data disagree.
        live_missing_rate = float(column.isna().mean())
        missing_delta = live_missing_rate - baseline.missing_rate
        status = classify(psi)
        if abs(missing_delta) >= MISSING_RATE_ALERT:
            status = "ALERT"
        elif abs(missing_delta) >= MISSING_RATE_WARN and status == "OK":
            status = "WARN"

        detail = (
            f"live mean {float(np.mean(values)):.3f} vs baseline {baseline.mean:.3f}; "
            f"live p50 {float(np.median(values)):.3f} vs baseline {baseline.p50:.3f}"
        )
        if abs(missing_delta) >= MISSING_RATE_WARN:
            detail += (
                f"; missing {baseline.missing_rate:.0%} -> {live_missing_rate:.0%} "
                f"- train/serve skew, the PSI above compares two different populations"
            )

        results.append(
            FeatureDrift(
                feature=name,
                kind="numeric",
                psi=psi,
                jensen_shannon=jensen_shannon_distance(baseline.frequencies, current_frequencies),
                status=status,
                n_current=int(values.size),
                new_categories=[],
                detail=detail,
            )
        )

    for name, categorical_baseline in profile.categorical.items():
        if name not in current.columns:
            results.append(_missing_feature(name, "categorical"))
            continue
        values = current[name].dropna().astype(str).tolist()
        if not values:
            results.append(_missing_feature(name, "categorical"))
            continue
        reference_counts, current_counts, unseen = align_categorical(
            categorical_baseline.frequencies, values
        )
        psi = population_stability_index(reference_counts, current_counts)
        results.append(
            FeatureDrift(
                feature=name,
                kind="categorical",
                psi=psi,
                jensen_shannon=jensen_shannon_distance(reference_counts, current_counts),
                status=classify(psi),
                n_current=len(values),
                new_categories=unseen,
                detail=(
                    f"{len(unseen)} level(s) unseen in training" if unseen else "no unseen levels"
                ),
            )
        )

    return sorted(results, key=lambda d: d.psi, reverse=True)


def _missing_feature(name: str, kind: str) -> FeatureDrift:
    return FeatureDrift(
        feature=name,
        kind=kind,
        psi=float("nan"),
        jensen_shannon=float("nan"),
        status="ALERT",
        n_current=0,
        new_categories=[],
        detail="feature absent from the live window - the serving contract has changed",
    )


def compute_prediction_drift(
    profile: ReferenceProfile, predictions: pd.Series
) -> tuple[float, str]:
    """Has the mix of predicted segments moved away from the training mix?

    The fastest signal of the four and the only one that needs neither labels
    nor per-feature work, which is why it is worth reporting on its own: a model
    whose output distribution has shifted while its inputs look stable is
    usually being fed a feature it cannot see moving.
    """
    reference_counts, current_counts, _ = align_categorical(
        profile.target_frequencies, predictions.astype(str).tolist()
    )
    psi = population_stability_index(reference_counts, current_counts)
    return psi, classify(psi)


def compute_confidence_drift(
    profile: ReferenceProfile, frame: pd.DataFrame, relative_tolerance: float
) -> dict[str, Any]:
    """The unsupervised proxy, used while a window is still unlabelled.

    A drop in mean confidence, or in the gap to the runner-up class, says the
    decision boundary no longer sits cleanly between the classes the traffic
    now contains. It is a proxy and is labelled as one: confidence can also
    fall for benign reasons (a genuinely harder month), which is exactly why it
    never escalates past WARN on its own.
    """
    baseline = profile.performance
    if baseline is None or baseline.mean_confidence is None:
        return {"status": "SKIPPED", "reason": "no confidence baseline in the reference profile"}

    live = frame["confidence"].dropna()
    if live.empty:
        return {"status": "SKIPPED", "reason": "no confidence recorded in this window"}

    live_confidence = float(live.mean())
    drop = baseline.mean_confidence - live_confidence
    relative_drop = drop / baseline.mean_confidence if baseline.mean_confidence else 0.0

    live_margin = None
    baseline_margin = baseline.mean_margin
    if "margin" in frame.columns and not frame["margin"].dropna().empty:
        live_margin = float(frame["margin"].dropna().mean())

    return {
        "status": "WARN" if relative_drop >= relative_tolerance else "OK",
        "baseline_mean_confidence": baseline.mean_confidence,
        "live_mean_confidence": live_confidence,
        "relative_drop": relative_drop,
        "baseline_mean_margin": baseline_margin,
        "live_mean_margin": live_margin,
        "reason": (
            f"mean confidence fell {relative_drop:.1%} versus baseline"
            if relative_drop >= relative_tolerance
            else "confidence stable"
        ),
    }


def compute_concept_drift(
    profile: ReferenceProfile,
    labelled: pd.DataFrame,
    min_labelled: int,
    warn_tolerance: float,
    alert_tolerance: float,
    bootstrap_resamples: int,
) -> ConceptDrift:
    """The real thing: has P(y|X) moved?

    Measured the only way it can be - by scoring live predictions against
    reconciled ground truth and comparing the result to the model's own held-out
    baseline. Two conditions must both hold before this escalates to ALERT:

      * **material** - the F1 drop is at least `alert_tolerance`, a number
        chosen because it is worth acting on, not because it is detectable;
      * **conclusive** - the whole bootstrap confidence interval for live F1
        sits below the baseline, so the drop is not an artefact of a small or
        unlucky labelled sample.

    A drop that is material but not conclusive is a WARN: the right response is
    to wait for more labels, not to pull the model. Reporting it as an alert
    would train the on-call to ignore alerts, which costs more than the drift.
    """
    baseline = profile.performance
    if baseline is None:
        return ConceptDrift(
            status="SKIPPED",
            n_labelled=int(len(labelled)),
            baseline_f1=None,
            live_f1=None,
            live_f1_lower=None,
            live_f1_upper=None,
            f1_drop=None,
            live_accuracy=None,
            statistically_significant=False,
            detail="the served model version has no performance baseline; retrain to publish one",
        )

    n_labelled = int(len(labelled))
    if n_labelled < min_labelled:
        return ConceptDrift(
            status="SKIPPED",
            n_labelled=n_labelled,
            baseline_f1=baseline.f1_weighted,
            live_f1=None,
            live_f1_lower=None,
            live_f1_upper=None,
            f1_drop=None,
            live_accuracy=None,
            statistically_significant=False,
            detail=(
                f"{n_labelled} labelled prediction(s) in the window, "
                f"{min_labelled} required - waiting for reconciliation"
            ),
        )

    y_true = labelled["actual_segment"].astype(str).to_numpy()
    y_pred = labelled["predicted_segment"].astype(str).to_numpy()

    def weighted_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
        return float(f1_score(truth, predicted, average="weighted", zero_division=0))

    interval: PerformanceInterval = bootstrap_metric_ci(
        y_true, y_pred, weighted_f1, n_resamples=bootstrap_resamples
    )
    drop = baseline.f1_weighted - interval.estimate
    significant = interval.upper < baseline.f1_weighted

    if drop >= alert_tolerance and significant:
        status = "ALERT"
        detail = (
            f"weighted F1 fell {drop:.4f} below the {baseline.f1_weighted:.4f} baseline, "
            f"and the whole 95% interval [{interval.lower:.4f}, {interval.upper:.4f}] "
            f"sits below it - the model's relationship to its target has changed"
        )
    elif drop >= alert_tolerance:
        status = "WARN"
        detail = (
            f"weighted F1 is {drop:.4f} below baseline, but the 95% interval "
            f"[{interval.lower:.4f}, {interval.upper:.4f}] still reaches it - "
            f"{n_labelled} labels is not yet enough to call it"
        )
    elif drop >= warn_tolerance:
        status = "WARN"
        detail = f"weighted F1 is {drop:.4f} below baseline - within tolerance, worth watching"
    else:
        status = "OK"
        detail = f"weighted F1 within {abs(drop):.4f} of baseline"

    per_class = {}
    for segment in sorted(set(y_true.tolist())):
        mask_true = y_true == segment
        mask_pred = y_pred == segment
        per_class[segment] = float(
            f1_score(mask_true, mask_pred, average="binary", zero_division=0)
        )

    return ConceptDrift(
        status=status,
        n_labelled=n_labelled,
        baseline_f1=baseline.f1_weighted,
        live_f1=interval.estimate,
        live_f1_lower=interval.lower,
        live_f1_upper=interval.upper,
        f1_drop=drop,
        live_accuracy=float(accuracy_score(y_true, y_pred)),
        statistically_significant=significant,
        per_class_f1=per_class,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------


def count_consecutive_alerts(
    client: MlflowClient, experiment_id: str, model_version: str | None
) -> int:
    """How many of the most recent windows for this model version already alerted.

    Read back out of MLflow rather than kept in a table of its own: the previous
    verdicts are already recorded there as tags, so a second store would be a
    second thing to keep consistent for no new information. Restricted to the
    same model version - a new version resets the count, because the evidence
    against the old one says nothing about the new one.
    """
    if model_version is None:
        return 0
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_version = '{model_version}'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    streak = 0
    for run in runs:
        if run.data.tags.get("drift_status") == "ALERT":
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_monitor(config: dict | None = None) -> DriftReport:
    config = config or load_config()
    monitoring_config = config["monitoring"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    registry_name = config["model"]["registry_name"]
    registry_alias = config["model"]["registry_alias"]

    version = resolve_served_version(client, registry_name, registry_alias)
    log.info(
        "served_model_resolved",
        registry_name=registry_name,
        alias=registry_alias,
        version=version.version,
        run_id=version.run_id,
    )

    window = Window.trailing(hours=float(monitoring_config["window_hours"]))
    store = MonitoringStore()
    coverage = store.coverage(window, model_version=str(version.version))

    experiment = mlflow.set_experiment(monitoring_config["experiment_name"])

    try:
        profile = load_reference_profile(version.run_id)
    except Exception as e:  # noqa: BLE001
        # A model trained before baselining existed. Report it as a gap in
        # coverage rather than a crash: the fix is a retrain, and an operator
        # needs to be told that, not handed a stack trace at 03:00.
        log.error("reference_profile_missing", run_id=version.run_id, error=str(e))
        report = _skipped_report(
            reason=(
                f"model version {version.version} carries no {REFERENCE_ARTIFACT_PATH}; "
                "retrain to publish a monitoring baseline"
            ),
            version=version,
            registry_name=registry_name,
            registry_alias=registry_alias,
            window=window,
            coverage=coverage,
        )
        _record_run(report, experiment.experiment_id, monitoring_config)
        return report

    min_predictions = int(monitoring_config["min_predictions"])
    if coverage["n_predictions"] < min_predictions:
        report = _skipped_report(
            reason=(
                f"{coverage['n_predictions']} prediction(s) in {window}, "
                f"{min_predictions} required - too little traffic to decide"
            ),
            version=version,
            registry_name=registry_name,
            registry_alias=registry_alias,
            window=window,
            coverage=coverage,
        )
        _record_run(report, experiment.experiment_id, monitoring_config)
        return report

    max_rows = int(monitoring_config["max_rows"])
    current = store.predictions(window, model_version=str(version.version), max_rows=max_rows)
    labelled = store.labelled_predictions(
        window, model_version=str(version.version), max_rows=max_rows
    )

    data_drift = compute_data_drift(profile, current)
    scored = [d for d in data_drift if not np.isnan(d.psi)]
    drift_share = (
        sum(1 for d in data_drift if d.status == "ALERT") / len(data_drift) if data_drift else 0.0
    )

    prediction_psi, prediction_status = compute_prediction_drift(
        profile, current["predicted_segment"]
    )
    confidence_drift = compute_confidence_drift(
        profile, current, float(monitoring_config["confidence_relative_tolerance"])
    )
    concept_drift = compute_concept_drift(
        profile,
        labelled,
        min_labelled=int(monitoring_config["min_labelled"]),
        warn_tolerance=float(monitoring_config["f1_warn_tolerance"]),
        alert_tolerance=float(monitoring_config["f1_alert_tolerance"]),
        bootstrap_resamples=int(monitoring_config["bootstrap_resamples"]),
    )

    # Data drift only escalates when a *share* of the feature space has moved.
    # One drifted feature out of thirty is normal seasonal noise; a third of
    # them moving together is a pipeline change or a genuinely different
    # population, and that is what the share threshold is calibrated for.
    data_status = "OK"
    if drift_share >= float(monitoring_config["drift_share_alert"]):
        data_status = "ALERT"
    elif drift_share >= float(monitoring_config["drift_share_warn"]) or any(
        d.status == "ALERT" for d in scored
    ):
        data_status = "WARN"
    if any(d.n_current == 0 for d in data_drift):
        data_status = "ALERT"  # a feature stopped arriving: a contract break

    status = _worst(
        [
            data_status,
            prediction_status if prediction_status != "ALERT" else "WARN",
            confidence_drift.get("status", "SKIPPED"),
            concept_drift.status,
        ]
    )

    # Only concept drift may reach ALERT on its own. Everything else is an
    # observation about the model's inputs or its confidence, and neither is
    # evidence that the model is wrong - promoting them to ALERT is how a
    # monitor ends up paging someone for a seasonal shift in booking mix.
    if status == "ALERT" and concept_drift.status != "ALERT" and data_status != "ALERT":
        status = "WARN"

    consecutive = count_consecutive_alerts(client, experiment.experiment_id, str(version.version))
    if status == "ALERT" and consecutive + 1 < int(
        monitoring_config["consecutive_alerts_required"]
    ):
        status = "WARN"
        escalation_note = (
            f"alert conditions met, but {int(monitoring_config['consecutive_alerts_required'])} "
            f"consecutive windows are required to escalate (this is #{consecutive + 1})"
        )
    else:
        escalation_note = ""

    reason = _summarise(
        data_status,
        drift_share,
        prediction_status,
        confidence_drift,
        concept_drift,
        escalation_note,
    )

    report = DriftReport(
        status=status,
        reason=reason,
        model_name=registry_name,
        model_alias=registry_alias,
        model_version=str(version.version),
        window_start=window.start.isoformat(),
        window_end=window.end.isoformat(),
        coverage=_stringify_coverage(coverage),
        data_drift=data_drift,
        drift_share=drift_share,
        prediction_drift_psi=prediction_psi,
        prediction_drift_status=prediction_status,
        confidence_drift=confidence_drift,
        concept_drift=concept_drift,
        consecutive_alerts=consecutive,
        generated_at=datetime.now(UTC).isoformat(),
    )

    _record_run(report, experiment.experiment_id, monitoring_config)
    return report


def _stringify_coverage(coverage: dict) -> dict:
    """Timestamps are not JSON-serialisable and end up in the run artifact."""
    return {
        key: (value.isoformat() if hasattr(value, "isoformat") else value)
        for key, value in coverage.items()
    }


def _skipped_report(
    reason: str, version, registry_name: str, registry_alias: str, window: Window, coverage: dict
) -> DriftReport:
    return DriftReport(
        status="SKIPPED",
        reason=reason,
        model_name=registry_name,
        model_alias=registry_alias,
        model_version=str(version.version),
        window_start=window.start.isoformat(),
        window_end=window.end.isoformat(),
        coverage=_stringify_coverage(coverage),
        data_drift=[],
        drift_share=0.0,
        prediction_drift_psi=None,
        prediction_drift_status="SKIPPED",
        confidence_drift={"status": "SKIPPED", "reason": reason},
        concept_drift=ConceptDrift(
            status="SKIPPED",
            n_labelled=int(coverage.get("n_labelled", 0)),
            baseline_f1=None,
            live_f1=None,
            live_f1_lower=None,
            live_f1_upper=None,
            f1_drop=None,
            live_accuracy=None,
            statistically_significant=False,
            detail=reason,
        ),
        consecutive_alerts=0,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _summarise(
    data_status: str,
    drift_share: float,
    prediction_status: str,
    confidence_drift: dict,
    concept_drift: ConceptDrift,
    escalation_note: str,
) -> str:
    parts = []
    if concept_drift.status in ("WARN", "ALERT"):
        parts.append(f"concept drift ({concept_drift.status}): {concept_drift.detail}")
    if data_status != "OK":
        parts.append(f"data drift ({data_status}): {drift_share:.0%} of features drifted")
    if prediction_status != "OK":
        parts.append(f"prediction drift ({prediction_status})")
    if confidence_drift.get("status") == "WARN":
        parts.append(f"confidence drift: {confidence_drift.get('reason')}")
    if escalation_note:
        parts.append(escalation_note)
    if not parts:
        parts.append("no drift detected against the served version's training baseline")
    return "; ".join(parts)


def _record_run(report: DriftReport, experiment_id: str, monitoring_config: dict) -> None:
    """Persist the whole verdict as an MLflow run.

    MLflow, not a metrics stack: these are four numbers a day about a model,
    already tied to the model version and the run that produced it, and MLflow
    is already this project's system of record for exactly that. Standing up
    Prometheus and Grafana to hold them would be a second observability plane
    to operate for no question it could answer that this cannot.
    """
    run_name = f"drift-{report.window_end}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(
            {
                "drift_status": report.status,
                "model_name": report.model_name,
                "model_alias": report.model_alias,
                "model_version": report.model_version or "unknown",
                "concept_drift_status": report.concept_drift.status,
                "data_drift_status": _worst([d.status for d in report.data_drift] or ["SKIPPED"]),
                "git_commit_hash": get_git_commit_hash(),
                "window_start": report.window_start,
                "window_end": report.window_end,
            }
        )

        metrics: dict[str, float] = {
            "n_predictions": float(report.coverage.get("n_predictions", 0)),
            "n_labelled": float(report.coverage.get("n_labelled", 0)),
            "label_coverage": float(report.coverage.get("label_coverage", 0.0)),
            "drift_share": report.drift_share,
        }
        if report.prediction_drift_psi is not None:
            metrics["prediction_drift_psi"] = report.prediction_drift_psi
        for key in ("live_mean_confidence", "baseline_mean_confidence", "relative_drop"):
            value = report.confidence_drift.get(key)
            if isinstance(value, (int, float)):
                metrics[f"confidence_{key}"] = float(value)
        for key in (
            "baseline_f1",
            "live_f1",
            "live_f1_lower",
            "live_f1_upper",
            "f1_drop",
            "live_accuracy",
        ):
            value = getattr(report.concept_drift, key)
            if value is not None:
                metrics[f"concept_{key}"] = float(value)
        # Per-feature PSI is logged so the MLflow UI can plot one feature's
        # drift across every window - the view that turns "PSI is 0.3 today"
        # into "PSI has been climbing for a week".
        for drift in report.data_drift:
            if not np.isnan(drift.psi):
                metrics[f"psi_{drift.feature}"] = drift.psi

        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "window_hours": monitoring_config["window_hours"],
                "min_predictions": monitoring_config["min_predictions"],
                "min_labelled": monitoring_config["min_labelled"],
                "f1_alert_tolerance": monitoring_config["f1_alert_tolerance"],
                "consecutive_alerts_required": monitoring_config["consecutive_alerts_required"],
            }
        )

        # Staged through a temp dir rather than the working directory: the
        # CronJob's container runs with a read-only root filesystem (the same
        # hardening every other workload here uses), and /tmp is the one
        # writable emptyDir it gets.
        with tempfile.TemporaryDirectory() as staging_dir:
            artifacts = Path(staging_dir)
            (artifacts / "drift_report.json").write_text(
                json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
            )
            (artifacts / "drift_report.html").write_text(
                render_html_report(report), encoding="utf-8"
            )
            mlflow.log_artifacts(str(artifacts), artifact_path="drift")

        log.info(
            "drift_run_recorded",
            run_id=run.info.run_id,
            status=report.status,
            reason=report.reason,
        )


def main() -> int:
    config = load_config()
    report = run_monitor(config)

    # One structured line carrying the whole verdict: this is what CloudWatch
    # Logs Insights filters on, and what a `kubectl logs` on the CronJob pod
    # shows first.
    log.info(
        "drift_report",
        status=report.status,
        reason=report.reason,
        model_version=report.model_version,
        window_start=report.window_start,
        window_end=report.window_end,
        n_predictions=report.coverage.get("n_predictions"),
        n_labelled=report.coverage.get("n_labelled"),
        concept_drift_status=report.concept_drift.status,
        drift_share=report.drift_share,
    )

    # A non-zero exit marks the Job failed, which is the signal Kubernetes
    # already surfaces everywhere - `kubectl get cronjob`, the events stream,
    # any alerting wired to job failures - without this project having to own an
    # alerting channel of its own. Same principle as train.py's QualityGateError:
    # a bad outcome must fail loudly rather than be logged and forgotten.
    if report.status == "ALERT" and config["monitoring"].get("fail_on_alert", True):
        log.error("drift_alert", reason=report.reason)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
