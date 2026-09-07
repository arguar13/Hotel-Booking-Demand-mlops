"""Near-real-time drift early-warning, layered on top of the daily CronJob.

api/monitoring.py's own docstring explains why Postgres, not Kafka, is the
prediction log: at-most-once delivery to an unbounded topic is the wrong
trade for the durable, joinable record concept drift's ground-truth
reconciliation depends on. This module does not disagree with that, and
does not try to replace it - it uses Kafka for exactly one thing: as a
*trigger*, telling this process "enough new predictions have landed, go
look", never as the transport for the data itself. The data this module
analyses is read from the same Postgres table drift_monitor.py reads,
through the same store.py, using the exact same statistics
(src/monitoring/statistics.py) and the exact same reference profile. The
only thing genuinely different here is *when* it looks: every
trigger_batch_size events or trigger_max_interval_seconds, whichever comes
first, instead of once a day.

Concretely, this is `run_monitor()` (drift_monitor.py) called with
`monitoring.stream`'s config overlaid on `monitoring`'s - a 15-minute
window, a much lower min_predictions floor, and its own MLflow experiment so
its hysteresis counter (count_consecutive_alerts) never mixes with the
batch job's. Concept drift is not special-cased out of that call: a
15-minute window will almost always have too few reconciled labels and
compute_concept_drift will correctly report SKIPPED, exactly as it does for
the batch monitor early in a labelling delay. That is not a limitation of
this module, it is the honest answer - concept drift fundamentally cannot be
real-time for a signal that only exists after a booking is reconciled
against its channel of record, and pretending otherwise would be worse than
saying so. This module's genuine contribution is catching data, prediction,
and confidence drift - all three computable from the serving side alone -
hours before the batch job would see the same window at all.

A confirmed ALERT here reaches mitigation.trigger_retrain() exactly the way
the batch job's does (drift_monitor.py calls the same function); this
module adds no separate mitigation path, only a faster-firing trigger for
the one that already exists.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime

import structlog
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.config_loader import load_config
from src.monitoring.drift_monitor import run_monitor

log = structlog.get_logger("hotel_mlops.monitoring.stream_consumer")

# kubernetes/base/stream-consumer.yaml's livenessProbe execs a freshness
# check against this file. The same "dead man's switch" idea 612's
# CloudWatch heartbeat uses (batch_inference.py), adapted to a long-running
# Deployment instead of a scheduled Job: a consumer that stopped polling
# Kafka - wedged, not crashed - produces no error Kubernetes can see on its
# own, so this process has to say "I am still alive" itself.
# /tmp, not a config value someone chose freely: this Deployment runs with a
# read-only root filesystem (same hardening as every other workload here -
# see drift_monitor.py's own tempfile.TemporaryDirectory() staging comment),
# and /tmp is the one writable emptyDir it gets.
HEARTBEAT_PATH = os.getenv(
    "STREAM_CONSUMER_HEARTBEAT_PATH", "/tmp/stream_consumer_heartbeat"  # nosec B108
)

# One retry pause after a Kafka-level error, not a retry loop with no floor:
# without this, a broker blip would spin this process at 100% CPU issuing
# reconnect attempts as fast as the client library allows.
KAFKA_ERROR_BACKOFF_SECONDS = 5.0


def _touch_heartbeat() -> None:
    with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
        f.write(datetime.now(UTC).isoformat())


def stream_config(config: dict) -> dict:
    """Overlay `monitoring.stream` onto `monitoring` - see config.yaml's comment.

    Everything `monitoring.stream` does not list (drift_share_*,
    confidence_relative_tolerance, f1_*, bootstrap_resamples, min_labelled,
    max_rows) is inherited from the parent block unchanged, so the two
    monitors only ever diverge where they are meant to.
    """
    monitoring = dict(config["monitoring"])
    overrides = monitoring.pop("stream")
    monitoring.update(overrides)
    return {**config, "monitoring": monitoring}


def _build_consumer(effective_monitoring: dict) -> KafkaConsumer:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        raise RuntimeError(
            "KAFKA_BOOTSTRAP_SERVERS is not set. Unlike api/main.py's producer "
            "(optional - a missing broker just disables publishing), this consumer "
            "has no reason to run without one; kubernetes/base/stream-consumer.yaml "
            "should not be deployed until terraform/msk.tf has been applied."
        )
    return KafkaConsumer(
        effective_monitoring["kafka_topic"],
        bootstrap_servers=bootstrap_servers,
        security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
        group_id=effective_monitoring["kafka_consumer_group"],
        # Each message only ever triggers a re-evaluation; it is never
        # itself the analysed data (that lives durably in Postgres via
        # api/monitoring.py). Losing an offset commit on a crash just makes
        # the next window's trigger fire a little early - never a wrong
        # drift verdict - so the client's default at-least-once delivery is
        # already more guarantee than this needs.
        auto_offset_reset="latest",
        consumer_timeout_ms=int(effective_monitoring["trigger_max_interval_seconds"] * 1000),
    )


def run_forever(config: dict | None = None) -> None:
    config = config or load_config()
    effective_config = stream_config(config)
    monitoring = effective_config["monitoring"]

    consumer = _build_consumer(monitoring)
    log.info(
        "stream_consumer_started",
        topic=monitoring["kafka_topic"],
        window_hours=monitoring["window_hours"],
        trigger_batch_size=monitoring["trigger_batch_size"],
        trigger_max_interval_seconds=monitoring["trigger_max_interval_seconds"],
    )

    try:
        while True:
            _touch_heartbeat()
            pending = 0
            try:
                # Blocks until either trigger_batch_size messages have
                # arrived or consumer_timeout_ms elapses with none - the
                # micro-batch trigger described in this module's docstring.
                # A low-traffic window still gets evaluated (pending stays
                # 0, the for-loop simply exits on timeout), which is the
                # correct behaviour: run_monitor()'s own min_predictions
                # guard reports SKIPPED cheaply rather than this module
                # having to reimplement that judgment.
                for _ in consumer:
                    pending += 1
                    _touch_heartbeat()
                    if pending >= int(monitoring["trigger_batch_size"]):
                        break
            except KafkaError as e:
                log.warning("stream_consumer_kafka_error", error=str(e))
                time.sleep(KAFKA_ERROR_BACKOFF_SECONDS)
                continue

            try:
                report = run_monitor(effective_config)
                log.info(
                    "stream_window_evaluated",
                    status=report.status,
                    reason=report.reason,
                    n_predictions=report.coverage.get("n_predictions"),
                    triggering_messages=pending,
                )
            except Exception as e:  # noqa: BLE001
                # A failed evaluation must not kill the consumer - the next
                # trigger tries again. Same principle as api/main.py's Kafka
                # producer: a dependency failing here must never cascade
                # into this process going down.
                log.error("stream_window_evaluation_failed", error=str(e))
    finally:
        consumer.close()


def main() -> int:
    run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
