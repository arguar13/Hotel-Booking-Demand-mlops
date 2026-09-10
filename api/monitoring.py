"""Production inference logging - the write half of the drift-check loop.

Every prediction this service answers is persisted, with a stable correlation
id, to `monitoring.predictions` in the same RDS instance MLflow already uses.
A separate, far lower-volume path records the ground-truth segment once a
booking is reconciled against its channel of record
(`monitoring.booking_labels`).

`monitoring.predictions` (specifically its `features` column) is what
`core_ml/src/monitoring/drift_check.py` reads on a schedule to compare a
recent batch of traffic against the training-time reference profile.
`monitoring.booking_labels` is not read by that check today - it exists so a
future, more thorough check (comparing live accuracy against a held-out
baseline, not just feature means) has the ground truth to do that with,
without needing a new endpoint or a schema change.

Design constraints, in priority order:

1. **Serving never depends on this.** A dead database, a full disk or a slow
   write must not add latency to /predict and must never turn a successful
   prediction into an HTTP error. Records go onto a bounded in-memory queue
   with a non-blocking put, are flushed in batches by a background task, and
   every database call goes through a circuit breaker.
2. **Bounded memory.** The queue has a hard cap. When it is full the oldest
   record is dropped and counted: shedding observability data under pressure
   is correct behaviour, growing unbounded until the kubelet OOM-kills the
   pod is not.
3. **At-most-once, deliberately.** Losing a handful of rows on SIGTERM is
   acceptable for a statistical check whose smallest decision window is
   hundreds of rows. Buying at-least-once (an outbox table, a WAL, a broker)
   would put the serving path back into the dependency chain, which
   constraint 1 forbids.
"""

import logging
import os
import queue
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg2
import pybreaker
import structlog
from psycopg2 import extras, pool
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger("hotel_mlops.api.monitoring")

# Fixed, not configurable: this schema name is also hardcoded in
# core_ml/src/monitoring/drift_check.py's queries. Making it an env var
# would let the two halves disagree at runtime with no way to detect it, and
# would turn every statement below into dynamic SQL for no benefit.
SCHEMA = "monitoring"

MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Flush when either bound is hit, whichever comes first: the batch size keeps a
# busy service from holding rows longer than necessary, the interval keeps a
# quiet one from holding them indefinitely.
FLUSH_MAX_BATCH = int(os.getenv("MONITORING_FLUSH_MAX_BATCH", "200"))
FLUSH_INTERVAL_SECONDS = float(os.getenv("MONITORING_FLUSH_INTERVAL_SECONDS", "5"))
QUEUE_MAX_SIZE = int(os.getenv("MONITORING_QUEUE_MAX_SIZE", "10000"))

# The DDL is idempotent and applied by the writer at startup. This project has
# no migration tool, and introducing one for a two-table, append-only schema
# with a single writer would be more moving parts than the problem has.
SCHEMA_DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.predictions (
    prediction_id     UUID PRIMARY KEY,
    predicted_at      TIMESTAMPTZ      NOT NULL,
    model_name        TEXT             NOT NULL,
    model_alias       TEXT             NOT NULL,
    model_version     TEXT,
    predicted_segment TEXT             NOT NULL,
    confidence        DOUBLE PRECISION,
    margin            DOUBLE PRECISION,
    features          JSONB            NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_predictions_predicted_at
    ON {SCHEMA}.predictions (predicted_at DESC);
CREATE INDEX IF NOT EXISTS ix_predictions_model_version
    ON {SCHEMA}.predictions (model_version, predicted_at DESC);

CREATE TABLE IF NOT EXISTS {SCHEMA}.booking_labels (
    prediction_id  UUID PRIMARY KEY
        REFERENCES {SCHEMA}.predictions (prediction_id) ON DELETE CASCADE,
    actual_segment TEXT        NOT NULL,
    labeled_at     TIMESTAMPTZ NOT NULL,
    label_source   TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_booking_labels_labeled_at
    ON {SCHEMA}.booking_labels (labeled_at DESC);
"""  # SCHEMA is a module constant, never user input - no dynamic SQL here

_INSERT_PREDICTION = f"""
INSERT INTO {SCHEMA}.predictions
    (prediction_id, predicted_at, model_name, model_alias, model_version,
     predicted_segment, confidence, margin, features)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (prediction_id) DO NOTHING
"""  # nosec B608

# Two properties this statement has to have, both learned the hard way:
#
# 1. **Idempotent.** A reconciliation job is at-least-once by nature - it
#    re-reads a window and re-posts it - so a repeated label must be a
#    correction, not a duplicate-key error the caller has to special-case.
#    Hence ON CONFLICT ... DO UPDATE.
#
# 2. **Tolerant of a prediction that is not durable yet.** Predictions are
#    flushed asynchronously (that is what keeps /predict fast), so there is a
#    real window in which a caller holds a prediction_id that has not reached
#    the table - and with two API replicas, the pod handling /feedback may not
#    even be the pod holding that row in its queue. A plain INSERT would hit the
#    foreign key, and because psycopg2 batches share a transaction, one such row
#    would reject the entire batch. Joining against `predictions` instead means
#    unknown ids are skipped rather than fatal, `cur.rowcount` reports how many
#    genuinely landed, and the caller retries the difference. The foreign key
#    stays as the integrity backstop; it is no longer the control flow.
_UPSERT_LABELS = f"""
INSERT INTO {SCHEMA}.booking_labels
    (prediction_id, actual_segment, labeled_at, label_source)
SELECT v.prediction_id, v.actual_segment, v.labeled_at, v.label_source
FROM (VALUES %s) AS v (prediction_id, actual_segment, labeled_at, label_source)
JOIN {SCHEMA}.predictions p ON p.prediction_id = v.prediction_id
ON CONFLICT (prediction_id) DO UPDATE
    SET actual_segment = EXCLUDED.actual_segment,
        labeled_at     = EXCLUDED.labeled_at,
        label_source   = EXCLUDED.label_source
"""  # nosec B608

_LABEL_VALUES_TEMPLATE = "(%s::uuid, %s, %s::timestamptz, %s)"


def _dsn_from_env() -> str | None:
    """Build the connection string from env this Deployment already receives.

    DB_HOST/DB_PORT/POSTGRES_* come from the `mlops-config` ConfigMap and the
    ESO-materialized `mlops-secrets` Secret, both already consumed via envFrom -
    so turning inference logging on in production needs no new credential, no
    new secret and no new IAM grant.
    """
    host = os.getenv("DB_HOST")
    password = os.getenv("POSTGRES_PASSWORD")
    if not host or not password or host == "REPLACED_BY_OVERLAY":
        return None
    return (
        f"host={host} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'postgres')} "
        f"user={os.getenv('POSTGRES_USER', 'mlops_user')} "
        f"password={password} "
        f"connect_timeout={os.getenv('MONITORING_CONNECT_TIMEOUT', '5')} "
        f"application_name=hotel-mlops-api"
    )


class _PooledConnection:
    """Context manager that returns a connection to the pool, poisoned or not."""

    def __init__(self, owner: "InferenceLogger") -> None:
        self._owner = owner
        self._conn: Any = None

    def __enter__(self) -> Any:
        self._conn = self._owner.acquire()
        return self._conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._conn is None:
            return
        # A connection that raised may be sitting in a broken transaction, so
        # hand it back flagged for close rather than lending the same poisoned
        # session to the next caller.
        self._owner.release(self._conn, close=exc_type is not None)
        self._conn = None


class InferenceLogger:
    """Bounded, non-blocking, breaker-guarded writer for the prediction log."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn if dsn is not None else _dsn_from_env()
        self._pool: pool.ThreadedConnectionPool | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self._dropped = 0
        self._written = 0
        self._lock = threading.Lock()
        self._schema_ready = False
        # Bounded-failure posture: after 5 consecutive failures, stop calling a
        # dependency that has already proven it is down and re-probe every 30s
        # instead of on every flush. main.py surfaces the resulting
        # CircuitBreakerError as a 503, so callers get a fast, honest failure
        # instead of a hung request.
        self.breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)

    # -- lifecycle ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return MONITORING_ENABLED and self._dsn is not None

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _connect(self) -> pool.ThreadedConnectionPool:
        return pool.ThreadedConnectionPool(minconn=1, maxconn=4, dsn=self._dsn)

    def acquire(self) -> Any:
        if self._pool is None:
            self._pool = self._connect()
        return self._pool.getconn()

    def release(self, conn: Any, close: bool = False) -> None:
        if self._pool is not None:
            self._pool.putconn(conn, close=close)

    def connection(self) -> _PooledConnection:
        return _PooledConnection(self)

    def start(self) -> None:
        """Open the pool and apply the schema. Never raises.

        A failure here is not fatal: `enabled` stays true, the queue keeps
        accepting records, and the next flush retries the connection through
        the breaker. Inference logging degrading is not a reason to fail
        readiness - serving predictions is.
        """
        if not self.enabled:
            reason = (
                "MONITORING_ENABLED=false"
                if not MONITORING_ENABLED
                else "DB_HOST/POSTGRES_PASSWORD not set"
            )
            log.info("inference_logging_disabled", reason=reason)
            return
        try:
            self._ensure_schema()
            log.info("inference_logging_ready", schema=SCHEMA)
        except Exception as e:  # noqa: BLE001 - deliberately never fatal
            log.warning("inference_logging_start_failed", error=str(e))

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_DDL)
            conn.commit()
        self._schema_ready = True

    def close(self) -> None:
        """Drain what is queued, then release the pool. Best effort, bounded."""
        if not self.enabled:
            return
        try:
            self.flush()
        except Exception as e:  # noqa: BLE001
            log.warning("inference_logging_final_flush_failed", error=str(e))
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    # -- write path --------------------------------------------------------

    def record_prediction(
        self,
        *,
        prediction_id: str,
        model_name: str,
        model_alias: str,
        model_version: str | None,
        predicted_segment: str,
        confidence: float | None,
        margin: float | None,
        features: dict[str, Any],
    ) -> None:
        """Enqueue one prediction. O(1), never blocks, never raises."""
        if not self.enabled:
            return
        row = (
            prediction_id,
            datetime.now(UTC),
            model_name,
            model_alias,
            model_version,
            predicted_segment,
            confidence,
            margin,
            extras.Json(features),
        )
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Shed the oldest record rather than refuse the newest: drift
            # statistics describe the *current* window, so under sustained
            # pressure the freshest rows are the ones worth keeping.
            with self._lock:
                self._dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(row)
            except (queue.Empty, queue.Full):
                pass

    def flush(self) -> int:
        """Drain up to FLUSH_MAX_BATCH rows into Postgres. Never raises.

        Returns the number of rows written - 0 when there was nothing to write,
        or when the write failed and the batch was discarded.
        """
        if not self.enabled:
            return 0

        batch: list[tuple] = []
        while len(batch) < FLUSH_MAX_BATCH:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return 0

        try:
            self.breaker.call(self._write_batch, batch)
        except pybreaker.CircuitBreakerError:
            # Breaker open: the database has already proven it is unreachable.
            # Discard rather than requeue - requeueing would grow the queue
            # until it evicted genuinely new records, trading fresh data for
            # stale data that already failed to persist once.
            with self._lock:
                self._dropped += len(batch)
            log.warning(
                "inference_log_batch_skipped", reason="circuit breaker open", rows=len(batch)
            )
            return 0
        except (psycopg2.Error, OSError) as e:
            with self._lock:
                self._dropped += len(batch)
            log.warning("inference_log_batch_failed", error=str(e), rows=len(batch))
            return 0

        with self._lock:
            self._written += len(batch)
        return len(batch)

    def _write_batch(self, batch: list[tuple]) -> None:
        with self.connection() as conn:
            if not self._schema_ready:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA_DDL)
                self._schema_ready = True
            with conn.cursor() as cur:
                extras.execute_batch(cur, _INSERT_PREDICTION, batch, page_size=FLUSH_MAX_BATCH)
            conn.commit()

    # -- ground truth ------------------------------------------------------

    def record_labels(self, labels: list[tuple[str, str, str]]) -> int:
        """Persist ground-truth segments. Synchronous, unlike predictions.

        This is the low-volume, back-office half of the loop: a reconciliation
        job posts a batch and needs to know whether it landed, so buying that
        confirmation with a synchronous write is the right trade - this is not
        the latency-critical path constraint 1 protects.

        Returns how many rows actually landed, which can be fewer than were sent:
        a prediction still sitting in a flush queue is skipped rather than
        rejected (see _UPSERT_LABELS). The caller compares that count against
        what it sent and re-posts the batch, which is safe because the write is
        idempotent. Raises on an actual failure, so a lost label surfaces as a
        503 rather than being silently swallowed.
        """
        if not self.enabled:
            raise RuntimeError("inference logging is disabled; labels cannot be recorded")

        now = datetime.now(UTC)
        rows = [(pid, segment, now, source) for pid, segment, source in labels]

        def _write() -> int:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    extras.execute_values(
                        cur,
                        _UPSERT_LABELS,
                        rows,
                        template=_LABEL_VALUES_TEMPLATE,
                        page_size=500,
                    )
                    written = cur.rowcount
                conn.commit()
            return max(int(written), 0)

        return int(self.breaker.call(_write))

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What /health reports, so a degraded sink is visible without log-diving."""
        with self._lock:
            dropped, written = self._dropped, self._written
        return {
            "enabled": self.enabled,
            "queue_depth": self._queue.qsize(),
            "rows_written": written,
            "rows_dropped": dropped,
            "circuit_breaker_state": self.breaker.current_state,
        }


def new_prediction_id() -> str:
    """The join key between a served prediction and its eventual ground truth."""
    return str(uuid.uuid4())


def jsonable_features(features: dict[str, Any]) -> dict[str, Any]:
    """Coerce a feature mapping into something JSONB round-trips losslessly.

    Pydantic hands back native Python types here, but NaN is a float that
    json.dumps writes as the bare token `NaN` - valid JavaScript, invalid JSON,
    and rejected by Postgres' JSONB parser. Normalising it (and the infinities)
    to null keeps one unrepresentable value from failing an entire batch.
    """

    def _clean(value: Any) -> Any:
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value

    return {k: _clean(v) for k, v in features.items()}


# Module-level singleton, wired up by main.py's startup/shutdown hooks.
inference_logger = InferenceLogger()

logging.getLogger("psycopg2").setLevel(logging.WARNING)
