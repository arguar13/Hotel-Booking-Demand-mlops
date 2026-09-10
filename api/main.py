import asyncio
import logging
import os
from typing import Any

# MLflow's own HTTP client already retries with backoff (default 5
# attempts). Left alone, that would compound with the tenacity retry
# around _load_model_with_retry below - 3 outer attempts each paying
# mlflow's own multi-attempt backoff - turning "a few bounded seconds" into
# minutes. One outer retry layer with a known, bounded budget is what "no
# infinite failure loops" actually requires, so mlflow's internal layer is
# dialed down to a single fast-failing attempt here instead.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
import pybreaker
import structlog
from fastapi import FastAPI, HTTPException, Response, status
from mlflow.tracking import MlflowClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from monitoring import (
    FLUSH_INTERVAL_SECONDS,
    inference_logger,
    jsonable_features,
    new_prediction_id,
)
from schemas import BookingFeatures, LabelIngest, LabelIngestResponse, PredictionResponse

# Structured JSON logging: every line is a single JSON object (timestamp,
# level, event, and whatever key=value context each call site attaches),
# ready to ship to CloudWatch Logs / Elasticsearch without a separate log
# parser. Human-readable console logging has no place in a container that
# only ever gets read by a log aggregator.
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger("hotel_mlops.api")

app = FastAPI(
    title="Hotel Market Segmentation API",
    description="MLOps API for multiclass market segment classification",
    version="1.0.0",
)

# Global model variable
model = None

# Which registry *version* the alias currently resolves to. Recorded on every
# logged prediction so a drift finding can be attributed to a specific model
# version rather than to the moving target the alias is: promoting a new
# version mid-window would otherwise silently mix two models' predictions into
# one distribution and make the resulting statistic meaningless.
model_version: str | None = None

# The MLflow Model Registry is the single, immutable source of truth for the
# serving model - no local model.joblib fallback. Only a version that passed
# the training quality gate (see core_ml/src/train.py) is aliased, so the API
# only ever serves a model that cleared the bar. Uses the Model Registry
# alias API rather than the classic (deprecated) stages API.
MODEL_NAME = "HotelSegmentClassifier"
MODEL_ALIAS = os.getenv("MODEL_REGISTRY_ALIAS", "staging")

# How long to wait between attempts while there is still no model at all.
MODEL_RETRY_SECONDS = float(os.getenv("MODEL_RETRY_SECONDS", "15"))
# How often to re-read the alias once a model *is* loaded, to pick up a newly
# promoted version without a pod restart. 0 disables that (load once and stop).
MODEL_REFRESH_SECONDS = float(os.getenv("MODEL_REFRESH_SECONDS", "0"))


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def _load_model_with_retry(model_uri: str):
    """At most 3 attempts, capped exponential backoff - bounded, not infinite.

    A transient blip (registry mid-restart, brief network partition) self-heals
    without a full pod restart; a genuinely down registry still fails within
    seconds rather than hanging the container's startup indefinitely.
    """
    return mlflow.sklearn.load_model(model_uri)


def _resolve_model_version() -> str | None:
    """Ask the registry which version the served alias points at right now.

    Best effort by design: the API can serve perfectly well without knowing the
    version number, so a registry hiccup here degrades the *attribution* of a
    logged prediction, never the prediction itself.
    """
    try:
        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
    except Exception as e:  # noqa: BLE001
        log.warning("model_version_lookup_failed", alias=MODEL_ALIAS, error=str(e))
        return None


@app.on_event("startup")
def load_model():
    """
    Loads the model version currently aliased MODEL_ALIAS from the MLflow
    Model Registry. If the registry is unreachable or has no version under
    that alias, `model` stays None and /predict fails explicitly with a 503
    instead of silently serving a stale, untracked local artifact.
    """
    global model, model_version

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        log.info("model_load_attempt", model_uri=model_uri)
        model = _load_model_with_retry(model_uri)
        model_version = _resolve_model_version()
        log.info("model_load_succeeded", model_uri=model_uri, model_version=model_version)
    except Exception as e:
        log.error(
            "model_load_failed",
            model_uri=model_uri,
            error=str(e),
            note="will keep retrying in the background; /ready stays 503 until it loads",
        )


@app.on_event("startup")
async def start_model_loader() -> None:
    """Keep trying to load the model for as long as there isn't one.

    Without this, a single failed startup was permanent. Every rollout restarts
    the api and mlflow Deployments at the same time, so the api routinely comes
    up while the tracking server is still starting, exhausts the three bounded
    attempts above, and then serves 503 forever - with /health returning 200 the
    whole time, so Kubernetes sees a healthy pod and never restarts it. In
    practice that meant every deploy left the API dead until someone noticed.

    The retry loop below also picks up a newly promoted model version without a
    pod restart, once MODEL_REFRESH_SECONDS is set.
    """
    if model is not None and MODEL_REFRESH_SECONDS <= 0:
        return
    asyncio.create_task(_model_loader_loop())


async def _model_loader_loop() -> None:
    global model, model_version

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    while True:
        if model is None:
            delay = MODEL_RETRY_SECONDS
        elif MODEL_REFRESH_SECONDS > 0:
            delay = MODEL_REFRESH_SECONDS
        else:
            return  # loaded, and refreshing is disabled - nothing left to do
        await asyncio.sleep(delay)
        try:
            # to_thread: mlflow's loader is blocking, and this coroutine shares
            # the event loop that serves requests.
            loaded = await asyncio.to_thread(_load_model_with_retry, model_uri)
        except Exception as e:
            log.warning("model_load_retry_failed", model_uri=model_uri, error=str(e))
            continue
        was_missing = model is None
        model = loaded
        model_version = await asyncio.to_thread(_resolve_model_version)
        log.info(
            "model_load_succeeded" if was_missing else "model_refreshed",
            model_uri=model_uri,
            model_version=model_version,
        )


@app.on_event("startup")
async def start_inference_logger() -> None:
    """Open the prediction-log sink and start draining its queue.

    Needs no broker, only the RDS instance and credentials this pod already
    has. It is what makes the drift check possible, and it is the reason a
    prediction served in the cluster is a durable, joinable record instead of
    a log line on stdout.
    """
    await asyncio.to_thread(inference_logger.start)
    if inference_logger.enabled:
        asyncio.create_task(_inference_log_flush_loop())


async def _inference_log_flush_loop() -> None:
    """Batch-drain the prediction queue on a fixed interval, forever.

    Runs on the same event loop that serves requests, so the blocking database
    write is pushed to a worker thread. `flush()` never raises, so this loop
    cannot die and silently stop persisting predictions - the failure mode that
    would make a drift report quietly describe a shrinking, unrepresentative
    sample instead of reporting that it had no data.
    """
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
        await asyncio.to_thread(inference_logger.flush)


@app.on_event("shutdown")
async def stop_inference_logger() -> None:
    """Flush whatever is still queued before the pod goes away.

    Best effort within the termination grace period: at-most-once delivery is
    the deliberate trade (see monitoring.py), so a batch lost to a hard kill is
    acceptable - but throwing away a full queue on every routine rollout, when
    draining it costs one bounded write, would not be.
    """
    await asyncio.to_thread(inference_logger.close)


@app.get("/health")
def health():
    """
    Liveness probe target: is the process itself alive and serving?

    Deliberately still 200 with `model_loaded=False`. A missing model is not a
    reason to kill the container - the background loader is retrying, and a
    restart would only make it start over. Readiness is what gates traffic; see
    /ready.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_alias": MODEL_ALIAS,
        "model_version": model_version,
        # A degraded prediction sink must not fail readiness, but it silently
        # starves the drift check of data. `rows_dropped` climbing is the
        # signal that the next drift check will run on an unrepresentative
        # sample.
        "inference_logging": inference_logger.stats(),
    }


@app.get("/ready")
def ready(response: Response):
    """
    Readiness probe target: can this replica actually answer /predict?

    Returns 503 until a model is loaded, so Kubernetes keeps the pod out of the
    Service's endpoints instead of load-balancing requests onto a replica that
    can only answer 503. During a rollout that is the difference between a few
    seconds of one replica serving and clients seeing real errors.
    """
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "model not loaded yet"}
    return {"status": "ready", "model_alias": MODEL_ALIAS}


def _predict_with_confidence(
    estimator: Any, input_data: pd.DataFrame
) -> tuple[str, float | None, float | None]:
    """Predict one row, returning the label plus its two uncertainty signals.

    `confidence` is the winning class probability; `margin` is the gap to the
    runner-up. Both are the standard unsupervised early-warning signals for
    concept drift: ground truth arrives with a reconciliation delay, so for as
    long as a window is unlabelled these are the only evidence available that
    the decision boundary no longer fits the traffic. Confidence alone is not
    enough - a model can stay confident while flipping between two classes it
    can no longer separate, and only the margin shows that.

    One forward pass, not two: for every scikit-learn classifier `predict` is
    defined as `classes_[argmax(predict_proba)]`, so taking the argmax here is
    identical to calling `predict` and half the work. Estimators without
    `predict_proba` (or without `classes_`) fall back to a plain label rather
    than failing - degrading the monitoring signal, never the endpoint.
    """
    predict_proba = getattr(estimator, "predict_proba", None)
    classes = getattr(estimator, "classes_", None)
    if predict_proba is None or classes is None:
        return str(estimator.predict(input_data)[0]), None, None

    probabilities = predict_proba(input_data)[0]
    ranked = np.argsort(probabilities)[::-1]
    confidence = float(probabilities[ranked[0]])
    runner_up = float(probabilities[ranked[1]]) if len(ranked) > 1 else 0.0
    return str(classes[ranked[0]]), confidence, confidence - runner_up


@app.post("/predict", response_model=PredictionResponse)
def predict_segment(features: BookingFeatures):
    """
    Predicts the market segment based on booking features.

    The response carries a `prediction_id`: quote it back on POST /feedback once
    the booking's true segment is known, and this prediction becomes part of the
    logged sample the drift check reads.
    """
    estimator = model
    if estimator is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    feature_map = features.model_dump()
    try:
        input_data = pd.DataFrame([feature_map])
        predicted_segment, confidence, margin = _predict_with_confidence(estimator, input_data)
    except Exception as e:
        log.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    prediction_id = new_prediction_id()

    log.info(
        "prediction_succeeded",
        prediction_id=prediction_id,
        predicted_market_segment=predicted_segment,
        confidence=confidence,
        model_version=model_version,
    )

    # Non-blocking and non-fatal by construction: this is an in-memory
    # enqueue, flushed to Postgres in the background (see monitoring.py).
    inference_logger.record_prediction(
        prediction_id=prediction_id,
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=model_version,
        predicted_segment=predicted_segment,
        confidence=confidence,
        margin=margin,
        features=jsonable_features(feature_map),
    )

    return PredictionResponse(
        predicted_market_segment=predicted_segment,
        prediction_id=prediction_id,
        model_version=model_version,
        confidence=confidence,
        margin=margin,
    )


@app.post("/feedback", response_model=LabelIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_labels(labels: list[LabelIngest]):
    """Record the ground-truth segment for previously served predictions.

    This is the delayed-label half of the loop, and the only reason this system
    can measure *concept* drift (a change in P(y|X)) rather than merely data
    drift (a change in P(X)). In this domain the truth is knowable: a booking's
    market segment is settled when the reservation is reconciled against its
    channel of record, hours to days after the prediction was served. That is
    what a nightly reconciliation job posts here, in batches.

    Deliberately not a fire-and-forget enqueue like /predict: labels are
    low-volume, arrive from a batch job that can retry, and are worthless if
    silently dropped - a monitor computing accuracy over a sample that quietly
    lost a biased subset of its labels reports a number that is worse than no
    number. So the write is synchronous and a failure is a 503 the caller can
    act on. The upsert is idempotent, so retrying a whole batch is safe.
    """
    if not labels:
        return LabelIngestResponse(accepted=0, skipped=0)

    if not inference_logger.enabled:
        raise HTTPException(
            status_code=503,
            detail="Inference logging is disabled; there is no prediction log to label.",
        )

    # Drain this replica's pending predictions first. Predictions are persisted
    # asynchronously, so a caller that reconciles quickly can hold an id that has
    # not reached the table yet; flushing here closes that window for everything
    # this pod served. It cannot close it for a sibling replica, which is why
    # `skipped` exists and why the caller is expected to re-post.
    await asyncio.to_thread(inference_logger.flush)

    try:
        accepted = await asyncio.to_thread(
            inference_logger.record_labels,
            [
                (item.prediction_id, item.actual_market_segment, item.label_source)
                for item in labels
            ],
        )
    except pybreaker.CircuitBreakerError as e:
        log.warning("label_ingest_skipped", reason="circuit breaker open", count=len(labels))
        raise HTTPException(
            status_code=503, detail="Monitoring store unavailable; retry this batch."
        ) from e
    except (psycopg2.Error, OSError, RuntimeError) as e:
        log.error("label_ingest_failed", error=str(e), count=len(labels))
        raise HTTPException(status_code=503, detail="Could not record labels.") from e

    skipped = len(labels) - accepted
    if skipped:
        # Either the prediction is still queued somewhere, or the id was never
        # served at all. Both are the caller's cue to re-post the batch, which is
        # safe: the write is an idempotent upsert.
        log.info("label_ingest_partial", accepted=accepted, skipped=skipped)
    else:
        log.info("label_ingest_succeeded", accepted=accepted)
    return LabelIngestResponse(accepted=accepted, skipped=skipped)
