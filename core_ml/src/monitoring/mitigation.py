"""Bounded, cooldown-guarded automatic retraining.

drift_monitor.py's own module docstring says, correctly at the time it was
written: "It never retrains or promotes anything ... Closing the loop
automatically - drift fires, model retrains, alias moves - would let a
data-quality incident upstream promote a model trained on the incident, with
no human between the two."

That objection is not wrong, and this module does not silently override it.
What changed is that "closing the loop" no longer means "trust whatever the
retrain produces": a bad training run cannot reach production through this
path, because *promotion* is a separate, later decision
(core_ml/src/promote_model.py) gated on a canary comparison against the model
already serving - not on the mere existence of a new run. What this module
adds is narrower than the thing drift_monitor.py argued against: it may
*launch a training pipeline*; it can never make that pipeline's output serve
traffic. The concern the original docstring raised - an incident promoting a
model trained on itself - is still structurally impossible, enforced one
layer further down rather than by never retraining at all.

Three mechanisms bound the "automatic" part, each closing a specific way this
goes wrong:

1. **Cooldown.** No new trigger within `cooldown_hours` of the last one,
   regardless of which caller is asking. Without this, a batch ALERT
   (drift_monitor.py) and a stream ALERT (stream_consumer.py) minutes apart -
   each correctly hysteresis-gated on its own terms - would still fire two
   full retrains for the same underlying incident.
2. **A hard ceiling.** At most `max_auto_retrains` triggers in a rolling
   `max_auto_retrains_window_days`. A drift source a retrain cannot fix (an
   upstream data-quality break, not a genuine distribution shift) would
   otherwise retrigger every cooldown window forever. Hitting the ceiling
   fails loudly - logged at ERROR - and stops, rather than looping; the same
   "evidence, not a decision" posture drift_monitor.py already applies to
   detection now applies to mitigation's own limits.
3. **A bounded HTTP call.** The GitLab Pipeline Trigger request has a fixed
   timeout and is never retried in a loop. A trigger that cannot reach
   GitLab fails this one mitigation attempt (recorded as such) rather than
   blocking the caller - the monitor process - indefinitely.

State lives in `monitoring.mitigation_events`, a third table alongside the
two api/monitoring.py already owns. Same reasoning as that module's own DDL
comment: no migration tool for a project this size, so the DDL is idempotent
and applied by whichever process uses it first. This module owns this one
table the same way InferenceLogger owns `predictions`/`booking_labels`.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import psycopg2
import requests
import structlog

from src.monitoring.store import dsn_from_env

log = structlog.get_logger("hotel_mlops.monitoring.mitigation")

SCHEMA = "monitoring"

SCHEMA_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.mitigation_events (
    event_id      UUID PRIMARY KEY,
    triggered_at  TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL,
    reason        TEXT NOT NULL,
    model_version TEXT,
    pipeline_id   TEXT,
    outcome       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_mitigation_events_triggered_at
    ON {SCHEMA}.mitigation_events (triggered_at DESC);
"""  # SCHEMA is a module constant, never user input - no dynamic SQL here

_INSERT_EVENT = f"""
INSERT INTO {SCHEMA}.mitigation_events
    (event_id, triggered_at, source, reason, model_version, pipeline_id, outcome)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""  # nosec B608

_SELECT_SINCE = f"""
SELECT triggered_at, outcome
FROM {SCHEMA}.mitigation_events
WHERE triggered_at >= %s
ORDER BY triggered_at DESC
"""  # nosec B608


@dataclass
class MitigationOutcome:
    """What happened when a drift verdict asked this module to act."""

    triggered: bool
    reason: str
    pipeline_id: str | None = None


def _ensure_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_DDL)
    conn.commit()


def _events_since(conn: Any, since: datetime) -> list[tuple[datetime, str]]:
    with conn.cursor() as cur:
        cur.execute(_SELECT_SINCE, (since,))
        return list(cur.fetchall())


def _record_event(
    conn: Any,
    source: str,
    reason: str,
    model_version: str | None,
    pipeline_id: str | None,
    outcome: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_EVENT,
            (
                str(uuid.uuid4()),
                datetime.now(UTC),
                source,
                reason,
                model_version,
                pipeline_id,
                outcome,
            ),
        )
    conn.commit()


def _trigger_gitlab_pipeline(
    reason: str, source: str, model_version: str | None, mitigation_config: dict
) -> str:
    """POST to GitLab's Pipeline Trigger API. Bounded: one call, one timeout, no retry loop."""
    api_url = os.getenv(mitigation_config["gitlab_api_url_env"])
    project_path = os.getenv(mitigation_config["gitlab_project_path_env"])
    token = os.getenv(mitigation_config["gitlab_token_env"])
    ref = os.getenv(mitigation_config["gitlab_ref_env"], "main")

    if not api_url or not project_path or project_path == "REPLACED_BY_OVERLAY" or not token:
        raise RuntimeError(
            "GitLab trigger is not configured (GITLAB_API_URL / GITLAB_PROJECT_PATH / "
            "GITLAB_TRIGGER_TOKEN) - cannot launch an automatic retrain."
        )

    url = f"{api_url}/projects/{quote(project_path, safe='')}/trigger/pipeline"
    timeout_seconds = float(mitigation_config["trigger_timeout_seconds"])
    response = requests.post(
        url,
        data={
            "token": token,
            "ref": ref,
            "variables[TRIGGER_SOURCE]": source,
            "variables[TRIGGER_REASON]": reason[:500],
            "variables[TRIGGER_MODEL_VERSION]": model_version or "unknown",
            "variables[AUTO_RETRAIN]": "true",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return str(response.json()["id"])


def trigger_retrain(
    source: str,
    reason: str,
    model_version: str | None,
    config: dict,
    dsn: str | None = None,
) -> MitigationOutcome:
    """Ask mitigation to launch a retrain for a confirmed ALERT verdict.

    `source` identifies the caller ("batch" for drift_monitor.py, "stream"
    for stream_consumer.py) so the recorded event says which monitor asked,
    without the two monitors needing to coordinate anything beyond sharing
    this table.

    Returns a MitigationOutcome rather than raising for the expected
    non-triggering paths (cooldown active, ceiling reached, mitigation
    disabled) - those are ordinary outcomes a caller logs and moves on from,
    not failures. A GitLab API error is caught and recorded the same way:
    this function's contract is "tell me what happened", never "crash the
    monitor that called you".
    """
    mitigation_config = config["mitigation"]
    if not mitigation_config.get("enabled", True):
        return MitigationOutcome(triggered=False, reason="mitigation.enabled is false")

    conn = psycopg2.connect(dsn or dsn_from_env())
    try:
        _ensure_schema(conn)
        now = datetime.now(UTC)

        cooldown = timedelta(hours=float(mitigation_config["cooldown_hours"]))
        recent = _events_since(conn, now - cooldown)
        if recent:
            last_triggered_at, _ = recent[0]
            outcome = MitigationOutcome(
                triggered=False,
                reason=(
                    f"cooldown active - last mitigation event at "
                    f"{last_triggered_at.isoformat()}, {mitigation_config['cooldown_hours']}h "
                    "must pass before another automatic trigger"
                ),
            )
            log.info("mitigation_skipped_cooldown", source=source, reason=outcome.reason)
            return outcome

        window = timedelta(days=float(mitigation_config["max_auto_retrains_window_days"]))
        window_events = _events_since(conn, now - window)
        if len(window_events) >= int(mitigation_config["max_auto_retrains"]):
            ceiling_reason = (
                f"{len(window_events)} automatic retrain(s) already triggered in the last "
                f"{mitigation_config['max_auto_retrains_window_days']} day(s) "
                f"(limit {mitigation_config['max_auto_retrains']}) - refusing to trigger "
                "another automatically; a drift source a retrain cannot fix needs a human, "
                "not another retrain"
            )
            log.error("mitigation_ceiling_reached", source=source, drift_reason=reason)
            _record_event(conn, source, reason, model_version, None, "ceiling_reached")
            return MitigationOutcome(triggered=False, reason=ceiling_reason)

        try:
            pipeline_id = _trigger_gitlab_pipeline(reason, source, model_version, mitigation_config)
        except (requests.RequestException, RuntimeError) as e:
            log.error("mitigation_trigger_failed", source=source, error=str(e))
            _record_event(conn, source, reason, model_version, None, "failed")
            return MitigationOutcome(triggered=False, reason=f"GitLab trigger call failed: {e}")

        _record_event(conn, source, reason, model_version, pipeline_id, "triggered")
        log.info(
            "mitigation_retrain_triggered",
            source=source,
            reason=reason,
            model_version=model_version,
            pipeline_id=pipeline_id,
        )
        return MitigationOutcome(
            triggered=True, reason="retrain triggered", pipeline_id=pipeline_id
        )
    finally:
        conn.close()
