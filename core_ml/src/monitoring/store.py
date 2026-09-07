"""Read side of the monitoring store.

The write side lives in `api/monitoring.py` and owns the schema; this module
only ever issues SELECTs against it, with explicit column lists so a column
added on the writer's side can never silently change the shape of what the
monitor reads.

The two halves are separate Poetry projects that ship as separate images, so
nothing at import time can prove they agree on the table shape. That agreement
is enforced where it actually matters instead:
`integration-tests/tests/test_monitoring_schema.py` applies the API's real DDL
constant to a throwaway Postgres container and then runs these exact queries
against it, so a divergence fails CI rather than a 3 a.m. CronJob.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import psycopg2
import structlog

log = structlog.get_logger("hotel_mlops.monitoring.store")

SCHEMA = "monitoring"

# Newest-first with a hard cap, matching the ix_predictions_predicted_at index.
# The cap is not an optimisation, it is a guard: an unbounded SELECT over a
# window that unexpectedly contains ten million rows would have the CronJob
# OOM-killed, and a monitor that dies under load is a monitor that goes quiet
# exactly when something is happening.
_SELECT_PREDICTIONS = f"""
SELECT prediction_id, predicted_at, model_version, predicted_segment,
       confidence, margin, features
FROM {SCHEMA}.predictions
WHERE predicted_at >= %(window_start)s
  AND predicted_at <  %(window_end)s
  AND (%(model_version)s IS NULL OR model_version = %(model_version)s)
ORDER BY predicted_at DESC
LIMIT %(max_rows)s
"""  # nosec B608

# The join that makes concept drift measurable. Anchored on the *prediction's*
# timestamp rather than the label's: a window has to mean "the predictions
# served between these two instants", otherwise a backfill of old labels would
# be read as a sudden change in current performance.
_SELECT_LABELLED = f"""
SELECT p.prediction_id, p.predicted_at, p.model_version, p.predicted_segment,
       p.confidence, p.margin, p.features,
       l.actual_segment, l.labeled_at, l.label_source
FROM {SCHEMA}.predictions p
JOIN {SCHEMA}.booking_labels l USING (prediction_id)
WHERE p.predicted_at >= %(window_start)s
  AND p.predicted_at <  %(window_end)s
  AND (%(model_version)s IS NULL OR p.model_version = %(model_version)s)
ORDER BY p.predicted_at DESC
LIMIT %(max_rows)s
"""  # nosec B608

_SELECT_COVERAGE = f"""
SELECT count(*)                        AS n_predictions,
       count(l.prediction_id)          AS n_labelled,
       min(p.predicted_at)             AS first_prediction,
       max(p.predicted_at)             AS last_prediction
FROM {SCHEMA}.predictions p
LEFT JOIN {SCHEMA}.booking_labels l USING (prediction_id)
WHERE p.predicted_at >= %(window_start)s
  AND p.predicted_at <  %(window_end)s
  AND (%(model_version)s IS NULL OR p.model_version = %(model_version)s)
"""  # nosec B608


@dataclass
class Window:
    """The half-open interval [start, end) a report describes."""

    start: datetime
    end: datetime

    @classmethod
    def trailing(cls, hours: float, now: datetime | None = None) -> Window:
        end = now or datetime.now(UTC)
        return cls(start=end - timedelta(hours=hours), end=end)

    def as_params(self) -> dict:
        return {"window_start": self.start, "window_end": self.end}

    def __str__(self) -> str:
        return f"[{self.start.isoformat()}, {self.end.isoformat()})"


def dsn_from_env() -> str:
    """Same env contract as the API's writer, from the same ConfigMap/Secret.

    The CronJob consumes `mlops-config` and `mlops-secrets` with the identical
    envFrom block the api Deployment uses, so there is exactly one definition of
    "where the monitoring store is" and it lives in Git.
    """
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
        f"application_name=hotel-mlops-drift-monitor"
    )


class MonitoringStore:
    """Windowed reads over the prediction log. Read-only by construction."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or dsn_from_env()

    def _query(self, sql: str, params: dict) -> pd.DataFrame:
        with psycopg2.connect(self._dsn) as conn:
            # A monitoring job must never be able to change what it observes.
            # This is belt-and-braces next to the SELECT-only statements above,
            # but it turns "we only wrote SELECTs" from a review promise into a
            # property the database enforces.
            conn.set_session(readonly=True)
            return pd.read_sql_query(sql, conn, params=params)

    def coverage(self, window: Window, model_version: str | None = None) -> dict:
        """How much data the window actually holds, before anything is computed.

        Read first and reported even when the job stops here: "we could not
        decide, and here is why" is a legitimate and useful outcome, whereas a
        drift verdict computed on forty rows is worse than silence.
        """
        params = {**window.as_params(), "model_version": model_version}
        frame = self._query(_SELECT_COVERAGE, params)
        row = frame.iloc[0]
        n_predictions = int(row["n_predictions"] or 0)
        n_labelled = int(row["n_labelled"] or 0)
        return {
            "n_predictions": n_predictions,
            "n_labelled": n_labelled,
            "label_coverage": (n_labelled / n_predictions) if n_predictions else 0.0,
            "first_prediction": row["first_prediction"],
            "last_prediction": row["last_prediction"],
        }

    def predictions(
        self, window: Window, model_version: str | None = None, max_rows: int = 200_000
    ) -> pd.DataFrame:
        params = {**window.as_params(), "model_version": model_version, "max_rows": max_rows}
        frame = self._query(_SELECT_PREDICTIONS, params)
        log.info("prediction_window_loaded", rows=len(frame), window=str(window))
        return _explode_features(frame)

    def labelled_predictions(
        self, window: Window, model_version: str | None = None, max_rows: int = 200_000
    ) -> pd.DataFrame:
        params = {**window.as_params(), "model_version": model_version, "max_rows": max_rows}
        frame = self._query(_SELECT_LABELLED, params)
        log.info("labelled_window_loaded", rows=len(frame), window=str(window))
        return _explode_features(frame)


def _explode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lift the JSONB `features` column into real DataFrame columns.

    psycopg2 already hands JSONB back as a dict, so this is a reshape rather
    than a parse. Feature columns are prefixed nowhere and merged at the top
    level so the drift loop can address them by the same names the reference
    profile uses.
    """
    if frame.empty:
        return frame.drop(columns=["features"], errors="ignore")

    exploded = pd.json_normalize(frame["features"])
    exploded.index = frame.index
    # Metadata wins a name collision: a feature literally called
    # "predicted_segment" would otherwise overwrite the model's own output and
    # silently make prediction drift read as zero.
    overlapping = [c for c in exploded.columns if c in frame.columns]
    if overlapping:
        log.warning("feature_name_collision", columns=overlapping)
        exploded = exploded.drop(columns=overlapping)
    return pd.concat([frame.drop(columns=["features"]), exploded], axis=1)
