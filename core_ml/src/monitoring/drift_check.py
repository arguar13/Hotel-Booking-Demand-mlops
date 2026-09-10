"""Simple drift check: is a recent batch of predictions still close to the
data the model was trained on?

This favors the smallest check that is still useful to a small team over a
heavier monitoring setup (PSI, Jensen-Shannon, bootstrap confidence
intervals, hysteresis, an automatic retrain trigger):

    1. At training time (see train.py), compute the mean/std of every numeric
       feature and remember them as a JSON "reference profile", logged as an
       MLflow artifact next to the model.
    2. On a schedule (the CronJob in kubernetes/base/drift-monitor-cronjob.yaml)
       or by hand (`make drift-check`), read the last N hours of predictions
       that `api/monitoring.py` logged to Postgres, and compare each numeric
       feature's mean in that batch against the reference using a one-sample
       z-test (scipy.stats).
    3. Log the result to MLflow and to stdout. That's it: no automatic
       retraining, no cooldown windows, no message bus. A human reads the
       report and decides whether to retrain (`make train` + `make promote`).

Why a z-test and not something fancier: the reference profile only stores a
mean and a standard deviation per feature (kilobytes, not a full copy of the
training data), so a z-test - "how many standard errors is the new batch's
mean from the training mean?" - is the natural fit. `scipy.stats.norm` turns
that z-score into a p-value. A feature is flagged when the p-value is below
`alpha` (default 0.05), i.e. the shift is unlikely to be random noise.

Categorical features are not covered by this check - tracking their
distributions well needs a bit more machinery (e.g. a chi-square test against
stored category frequencies), which is a reasonable next step but is left out
here to keep the file easy to read end to end.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import structlog
from mlflow.tracking import MlflowClient
from scipy import stats

from src.config_loader import load_config

log = structlog.get_logger("hotel_mlops.monitoring")

REFERENCE_ARTIFACT_PATH = "monitoring/reference_profile.json"
SCHEMA = "monitoring"


# ---------------------------------------------------------------------------
# The reference profile: mean/std per numeric feature, built once at training
# time and logged as an MLflow artifact of that run (see train.py).
# ---------------------------------------------------------------------------


@dataclass
class NumericBaseline:
    mean: float
    std: float
    missing_rate: float


@dataclass
class PerformanceBaseline:
    """Held-out metrics at training time - what promote_model.py compares a
    new candidate against (see promote_model.py's module docstring)."""

    f1_weighted: float
    accuracy: float
    n_eval: int


@dataclass
class ReferenceProfile:
    created_at: str
    n_rows: int
    numeric: dict[str, NumericBaseline] = field(default_factory=dict)
    performance: PerformanceBaseline | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceProfile:
        performance = payload.get("performance")
        return cls(
            created_at=payload["created_at"],
            n_rows=int(payload["n_rows"]),
            numeric={k: NumericBaseline(**v) for k, v in payload.get("numeric", {}).items()},
            performance=PerformanceBaseline(**performance) if performance else None,
        )

    @classmethod
    def read(cls, path: str | Path) -> ReferenceProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_reference_profile(
    features: pd.DataFrame,
    performance: PerformanceBaseline | None = None,
) -> ReferenceProfile:
    """Summarise the numeric columns of the training data into mean/std pairs.

    `features` must be the columns the model actually consumes.
    """
    numeric: dict[str, NumericBaseline] = {}
    for column in features.select_dtypes(include=["number"]).columns:
        series = features[column]
        clean = series.dropna()
        if clean.empty:
            continue
        numeric[str(column)] = NumericBaseline(
            mean=float(clean.mean()),
            std=float(clean.std(ddof=0)) or 1e-9,  # avoid a zero-std z-test
            missing_rate=float(series.isna().mean()),
        )

    return ReferenceProfile(
        created_at=datetime.now(UTC).isoformat(),
        n_rows=int(len(features)),
        numeric=numeric,
        performance=performance,
    )


def load_reference_profile(run_id: str) -> ReferenceProfile:
    """Fetch the baseline logged alongside a specific model version's run."""
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=REFERENCE_ARTIFACT_PATH
    )
    return ReferenceProfile.read(local_path)


# ---------------------------------------------------------------------------
# The check itself: one z-test per numeric feature.
# ---------------------------------------------------------------------------


@dataclass
class FeatureCheck:
    feature: str
    reference_mean: float
    batch_mean: float
    z_score: float
    p_value: float
    status: str  # "OK" | "ALERT"


@dataclass
class DriftCheckResult:
    status: str  # "OK" | "ALERT" | "SKIPPED"
    reason: str
    checked_at: str
    n_rows: int
    features: list[FeatureCheck]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = [asdict(f) for f in self.features]
        return payload


def check_batch(
    profile: ReferenceProfile, batch: pd.DataFrame, alpha: float = 0.05
) -> DriftCheckResult:
    """One-sample z-test of each numeric feature's batch mean vs. the baseline.

        z = (batch_mean - reference_mean) / (reference_std / sqrt(n))

    A large |z| means the batch mean is many standard errors away from what
    training saw - unlikely to be chance. `scipy.stats.norm.sf` turns |z| into
    a two-sided p-value; a feature is flagged ALERT when that p-value drops
    below `alpha`.
    """
    checks: list[FeatureCheck] = []

    for name, baseline in profile.numeric.items():
        if name not in batch.columns:
            continue
        values = pd.to_numeric(batch[name], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size < 2:
            continue

        batch_mean = float(values.mean())
        standard_error = baseline.std / np.sqrt(values.size)
        z_score = (batch_mean - baseline.mean) / standard_error if standard_error else 0.0
        p_value = float(2 * stats.norm.sf(abs(z_score)))

        checks.append(
            FeatureCheck(
                feature=name,
                reference_mean=baseline.mean,
                batch_mean=batch_mean,
                z_score=float(z_score),
                p_value=p_value,
                status="ALERT" if p_value < alpha else "OK",
            )
        )

    if not checks:
        return DriftCheckResult(
            status="SKIPPED",
            reason="no numeric feature in the batch matched the reference profile",
            checked_at=datetime.now(UTC).isoformat(),
            n_rows=int(len(batch)),
            features=[],
        )

    alerted = [c.feature for c in checks if c.status == "ALERT"]
    status = "ALERT" if alerted else "OK"
    reason = (
        f"{len(alerted)} feature(s) shifted beyond alpha={alpha}: {', '.join(alerted)}"
        if alerted
        else "no feature mean shifted beyond the significance threshold"
    )

    return DriftCheckResult(
        status=status,
        reason=reason,
        checked_at=datetime.now(UTC).isoformat(),
        n_rows=int(len(batch)),
        features=checks,
    )


# ---------------------------------------------------------------------------
# Reading a recent batch of predictions out of Postgres. Same table
# `api/monitoring.py` writes to; this only ever runs SELECTs against it.
# ---------------------------------------------------------------------------


def dsn_from_env() -> str:
    host = os.getenv("DB_HOST")
    password = os.getenv("POSTGRES_PASSWORD")
    if not host or host == "REPLACED_BY_OVERLAY":
        raise RuntimeError("DB_HOST is not set; the monitoring store is unreachable.")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is not set; the monitoring store is unreachable.")
    return (
        f"host={host} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'postgres')} "
        f"user={os.getenv('POSTGRES_USER', 'mlops_user')} "
        f"password={password} "
        f"connect_timeout={os.getenv('MONITORING_CONNECT_TIMEOUT', '10')} "
        f"application_name=hotel-mlops-drift-check"
    )


_SELECT_RECENT_PREDICTIONS = f"""
SELECT features
FROM {SCHEMA}.predictions
WHERE predicted_at >= %(window_start)s
ORDER BY predicted_at DESC
LIMIT %(max_rows)s
"""  # nosec B608


def fetch_recent_predictions(
    window_hours: float, max_rows: int, dsn: str | None = None
) -> pd.DataFrame:
    """The last `window_hours` of logged predictions, features exploded into columns."""
    from datetime import timedelta

    window_start = datetime.now(UTC) - timedelta(hours=window_hours)
    params = {"window_start": window_start, "max_rows": max_rows}

    with psycopg2.connect(dsn or dsn_from_env()) as conn:
        conn.set_session(readonly=True)
        frame = pd.read_sql_query(_SELECT_RECENT_PREDICTIONS, conn, params=params)

    if frame.empty or "features" not in frame.columns:
        return pd.DataFrame()
    return pd.json_normalize(frame["features"])


# ---------------------------------------------------------------------------
# Orchestration: the CronJob entrypoint.
# ---------------------------------------------------------------------------


def run_check(config: dict | None = None) -> DriftCheckResult:
    config = config or load_config()
    monitoring_config = config["monitoring"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    registry_name = config["model"]["registry_name"]
    registry_alias = config["model"]["registry_alias"]
    version = client.get_model_version_by_alias(registry_name, registry_alias)
    run_id = version.run_id
    if not run_id:
        raise RuntimeError(
            f"model version {version.version} has no run_id on record; cannot load its "
            "reference profile"
        )

    try:
        profile = load_reference_profile(run_id)
    except Exception as e:  # noqa: BLE001
        log.error("reference_profile_missing", run_id=version.run_id, error=str(e))
        result = DriftCheckResult(
            status="SKIPPED",
            reason=f"model version {version.version} has no reference profile; retrain to publish one",
            checked_at=datetime.now(UTC).isoformat(),
            n_rows=0,
            features=[],
        )
        _record_run(result, monitoring_config)
        return result

    min_predictions = int(monitoring_config["min_predictions"])
    batch = fetch_recent_predictions(
        window_hours=float(monitoring_config["window_hours"]),
        max_rows=int(monitoring_config["max_rows"]),
    )
    if len(batch) < min_predictions:
        result = DriftCheckResult(
            status="SKIPPED",
            reason=f"{len(batch)} prediction(s) in the window, {min_predictions} required",
            checked_at=datetime.now(UTC).isoformat(),
            n_rows=len(batch),
            features=[],
        )
        _record_run(result, monitoring_config)
        return result

    result = check_batch(profile, batch, alpha=float(monitoring_config["alpha"]))
    _record_run(result, monitoring_config)
    return result


def _record_run(result: DriftCheckResult, monitoring_config: dict) -> None:
    """Log the verdict to MLflow - one run per check, same store as everything else."""
    experiment = mlflow.set_experiment(monitoring_config["experiment_name"])
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="drift-check") as run:
        mlflow.set_tags({"drift_status": result.status})
        mlflow.log_metric("n_rows", result.n_rows)
        mlflow.log_metric("n_features_checked", len(result.features))
        mlflow.log_metric(
            "n_features_alerted", sum(1 for f in result.features if f.status == "ALERT")
        )
        for feature in result.features:
            mlflow.log_metric(f"z_{feature.feature}", feature.z_score)
        mlflow.log_dict(result.to_dict(), "drift/drift_report.json")
        log.info(
            "drift_check_recorded",
            run_id=run.info.run_id,
            status=result.status,
            reason=result.reason,
        )


def main() -> int:
    config = load_config()
    result = run_check(config)

    log.info(
        "drift_check_result",
        status=result.status,
        reason=result.reason,
        n_rows=result.n_rows,
        alerted_features=[f.feature for f in result.features if f.status == "ALERT"],
    )

    if result.status == "ALERT" and config["monitoring"].get("fail_on_alert", True):
        log.error("drift_alert", reason=result.reason)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
