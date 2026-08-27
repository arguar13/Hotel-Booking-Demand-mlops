"""Builds the real api/Dockerfile (the exact image shipped to ECR/EKS) and
boots it as an ephemeral container, proving the container itself is sound
(dependencies resolve, uvicorn starts, the app responds) independently of
whether MLflow/Kafka happen to be reachable - MLflow is deliberately pointed
at a closed port so the test doesn't depend on the client's real retry timing.

Requires network access from inside the build for `poetry install`; on a
machine with a TLS-intercepting proxy/antivirus this build step needs that
proxy's CA trusted inside Docker (see README's "Local environment notes").
"""

from pathlib import Path

import requests
from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from testcontainers.core.waiting_utils import wait_for_logs

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_api_container_boots_and_serves_health() -> None:
    with DockerImage(
        path=str(REPO_ROOT), dockerfile_path="Dockerfile", tag="hotel-mlops-api:it"
    ) as image:
        container = (
            DockerContainer(str(image))
            .with_exposed_ports(8000)
            # Point at a closed local port instead of leaving the default
            # DNS name: an immediate connection-refused fails MLflow's
            # client fast, instead of waiting out its DNS-failure retry
            # backoff (which can run well past a minute).
            .with_env("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
            .with_env("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        )
        with container:
            wait_for_logs(container, "Application startup complete", timeout=30)

            host = container.get_container_host_ip()
            port = container.get_exposed_port(8000)

            health = requests.get(f"http://{host}:{port}/health", timeout=10)
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            openapi = requests.get(f"http://{host}:{port}/openapi.json", timeout=10)
            assert openapi.status_code == 200
