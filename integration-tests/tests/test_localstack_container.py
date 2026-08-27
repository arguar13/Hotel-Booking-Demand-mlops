"""Proves the exact S3 / SQS / Secrets Manager call patterns this project
uses in production (MLflow's S3 artifact store, DVC's S3 remote, and the
`aws secretsmanager get-secret-value` call in .gitlab-ci.yml's deploy-eks
job) work against a real emulated AWS - not just against mocks - before
they're ever tried against a real account.

Pinned to 4.14.0 for the same reason as docker-compose.yml's localstack
service: LocalStack >= 2026.03.0 requires a free-tier auth token.
"""

import json

import boto3
import pytest
from testcontainers.community.localstack import LocalStackContainer

LOCALSTACK_IMAGE = "localstack/localstack:4.14.0"


@pytest.fixture(scope="module")
def localstack():
    with LocalStackContainer(image=LOCALSTACK_IMAGE).with_services(
        "s3", "sqs", "secretsmanager"
    ) as container:
        yield container


def _client(localstack, service: str):
    return boto3.client(
        service,
        endpoint_url=localstack.get_url(),
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


def test_s3_bucket_round_trip(localstack) -> None:
    s3 = _client(localstack, "s3")
    s3.create_bucket(Bucket="mlflow-artifacts")

    s3.put_object(Bucket="mlflow-artifacts", Key="model/MLmodel", Body=b"flavor: sklearn")
    body = s3.get_object(Bucket="mlflow-artifacts", Key="model/MLmodel")["Body"].read()

    assert body == b"flavor: sklearn"


def test_sqs_queue_round_trip(localstack) -> None:
    sqs = _client(localstack, "sqs")
    queue_url = sqs.create_queue(QueueName="model-quality-alerts")["QueueUrl"]

    sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps({"run_id": "abc123", "f1": 0.4}))
    messages = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=5).get("Messages", [])

    assert len(messages) == 1
    assert json.loads(messages[0]["Body"]) == {"run_id": "abc123", "f1": 0.4}


def test_secrets_manager_round_trip(localstack) -> None:
    secretsmanager = _client(localstack, "secretsmanager")
    secretsmanager.create_secret(
        Name="hotel-mlops/db-password",
        SecretString=json.dumps({"db_password": "local-dev-password"}),
    )

    secret = secretsmanager.get_secret_value(SecretId="hotel-mlops/db-password")
    assert json.loads(secret["SecretString"]) == {"db_password": "local-dev-password"}
