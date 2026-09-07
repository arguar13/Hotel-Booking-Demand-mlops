from __future__ import annotations

import pytest

from src.monitoring import stream_consumer


def _config() -> dict:
    return {
        "monitoring": {
            "experiment_name": "batch-experiment",
            "window_hours": 24,
            "min_predictions": 500,
            "min_labelled": 300,
            "max_rows": 200000,
            "drift_share_warn": 0.20,
            "drift_share_alert": 0.35,
            "confidence_relative_tolerance": 0.10,
            "f1_warn_tolerance": 0.03,
            "f1_alert_tolerance": 0.05,
            "bootstrap_resamples": 1000,
            "consecutive_alerts_required": 2,
            "fail_on_alert": True,
            "stream": {
                "experiment_name": "stream-experiment",
                "window_hours": 0.25,
                "min_predictions": 50,
                "consecutive_alerts_required": 3,
                "kafka_topic": "predictions",
                "kafka_consumer_group": "hotel-mlops-stream-monitor",
                "trigger_batch_size": 25,
                "trigger_max_interval_seconds": 300,
            },
        }
    }


def test_stream_config_overlays_only_the_listed_fields() -> None:
    effective = stream_consumer.stream_config(_config())
    monitoring = effective["monitoring"]

    # Overridden by monitoring.stream.
    assert monitoring["experiment_name"] == "stream-experiment"
    assert monitoring["window_hours"] == 0.25
    assert monitoring["min_predictions"] == 50
    assert monitoring["consecutive_alerts_required"] == 3

    # Inherited unchanged from the parent monitoring block.
    assert monitoring["min_labelled"] == 300
    assert monitoring["drift_share_alert"] == 0.35
    assert monitoring["f1_alert_tolerance"] == 0.05
    assert monitoring["bootstrap_resamples"] == 1000

    # The nested "stream" key must not leak into the effective config -
    # run_monitor() indexes monitoring_config["window_hours"] etc. directly
    # and has no reason to see its own overlay source.
    assert "stream" not in monitoring


def test_stream_config_does_not_mutate_the_input() -> None:
    original = _config()
    stream_consumer.stream_config(original)
    assert "stream" in original["monitoring"]
    assert original["monitoring"]["experiment_name"] == "batch-experiment"


def test_build_consumer_requires_kafka_bootstrap_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    effective_monitoring = stream_consumer.stream_config(_config())["monitoring"]

    with pytest.raises(RuntimeError, match="KAFKA_BOOTSTRAP_SERVERS"):
        stream_consumer._build_consumer(effective_monitoring)
