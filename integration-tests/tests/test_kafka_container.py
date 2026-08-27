"""Proves the produce/consume pattern api/main.py's optional prediction-event
producer depends on works, against a real (ephemeral) Kafka broker.
"""

import json

from kafka import KafkaConsumer, KafkaProducer
from testcontainers.community.kafka import KafkaContainer


def test_kafka_container_produce_and_consume_roundtrip() -> None:
    with KafkaContainer("confluentinc/cp-kafka:7.6.0") as kafka:
        bootstrap_servers = kafka.get_bootstrap_server()

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        event = {"predicted_market_segment": "Direct", "run": "integration-test"}
        producer.send("predictions", event)
        producer.flush()

        consumer = KafkaConsumer(
            "predictions",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            consumer_timeout_ms=10000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        try:
            messages = [record.value for record in consumer]
        finally:
            consumer.close()

        assert event in messages
