import asyncio
import json
import logging
import os
import time

# MLflow's own HTTP client already retries with backoff (default 5
# attempts). Left alone, that would compound with the tenacity retry
# around _load_model_with_retry below - 3 outer attempts each paying
# mlflow's own multi-attempt backoff - turning "a few bounded seconds" into
# minutes. One outer retry layer with a known, bounded budget is what "no
# infinite failure loops" actually requires, so mlflow's internal layer is
# dialed down to a single fast-failing attempt here instead.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

import mlflow.sklearn
import pandas as pd
import pybreaker
import structlog
from fastapi import FastAPI, HTTPException, Response, status
from kafka import KafkaProducer
from kafka.errors import KafkaError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from schemas import BookingFeatures

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

# Publishing prediction events to Kafka is entirely optional: it is only
# attempted when KAFKA_BOOTSTRAP_SERVERS is set, and a Kafka outage must
# never take down model serving. This gives downstream consumers (drift
# monitoring, audit logging, feature stores) an async feed of predictions
# without coupling the request path's availability to Kafka's.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_PREDICTIONS_TOPIC = os.getenv("KAFKA_PREDICTIONS_TOPIC", "predictions")
_kafka_producer: KafkaProducer | None = None

# Once 5 consecutive publishes fail, stop even trying for 30s: without this,
# a downed Kafka broker would make every single /predict request pay a full
# connection-timeout on the publish call - a slow-motion, self-inflicted
# outage of the *serving* path caused by a dependency that isn't even on
# the critical path. This is the bounded-failure mechanism the "no infinite
# retry loops" requirement calls for, applied to the one place in this
# service that talks to an external system on every request.
_kafka_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)


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


@app.on_event("startup")
def load_model():
    """
    Loads the model version currently aliased MODEL_ALIAS from the MLflow
    Model Registry. If the registry is unreachable or has no version under
    that alias, `model` stays None and /predict fails explicitly with a 503
    instead of silently serving a stale, untracked local artifact.
    """
    global model

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        log.info("model_load_attempt", model_uri=model_uri)
        model = _load_model_with_retry(model_uri)
        log.info("model_load_succeeded", model_uri=model_uri)
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
    global model

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
        log.info(
            "model_load_succeeded" if was_missing else "model_refreshed",
            model_uri=model_uri,
        )


@app.on_event("startup")
def load_kafka_producer():
    """
    Connects the optional prediction-event producer. Never raises: if Kafka
    isn't configured or isn't reachable, `_kafka_producer` stays None and
    /predict simply skips publishing, logging a warning instead of failing.
    """
    global _kafka_producer

    if not KAFKA_BOOTSTRAP_SERVERS:
        log.info("kafka_producer_disabled", reason="KAFKA_BOOTSTRAP_SERVERS not set")
        return

    try:
        _kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=5000,
        )
        log.info("kafka_producer_connected", bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    except KafkaError as e:
        log.warning(
            "kafka_producer_connect_failed",
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            error=str(e),
        )


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
        "kafka_circuit_breaker_state": _kafka_breaker.current_state,
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


@app.post("/predict")
def predict_segment(features: BookingFeatures):
    """
    Predicts the market segment based on booking features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    try:
        input_data = pd.DataFrame([features.model_dump()])
        prediction = model.predict(input_data)
        predicted_segment = str(prediction[0])
    except Exception as e:
        log.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    log.info("prediction_succeeded", predicted_market_segment=predicted_segment)
    _publish_prediction_event(features, predicted_segment)
    return {"predicted_market_segment": predicted_segment}


def _publish_prediction_event(features: BookingFeatures, predicted_segment: str) -> None:
    """Best-effort publish; Kafka being down must never fail the HTTP response."""
    if _kafka_producer is None:
        return

    event = {
        "timestamp": time.time(),
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "features": features.model_dump(),
        "predicted_market_segment": predicted_segment,
    }
    try:
        _kafka_breaker.call(_kafka_producer.send, KAFKA_PREDICTIONS_TOPIC, event)
    except pybreaker.CircuitBreakerError:
        log.warning("kafka_publish_skipped", reason="circuit breaker open")
    except KafkaError as e:
        log.warning("kafka_publish_failed", error=str(e))
