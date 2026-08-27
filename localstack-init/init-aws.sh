#!/bin/bash
# Runs automatically once LocalStack's edge service is ready
# (LocalStack executes every *.sh file under /etc/localstack/init/ready.d/).
# Provisions the same AWS resources terraform/ provisions in real AWS, so
# the app and its tests can exercise the exact same S3/SQS/Secrets Manager
# calls locally before anything touches a real account.
set -euo pipefail

echo "==> Creating S3 buckets"
awslocal s3 mb s3://mlflow-artifacts
awslocal s3 mb s3://hotel-mlops-dvc-store

echo "==> Creating SQS queue (model-quality-alerts: quality-gate failures / retrain triggers)"
awslocal sqs create-queue --queue-name model-quality-alerts

echo "==> Creating Secrets Manager secret (mirrors hotel-mlops/db-password used in CI/deploy-eks)"
awslocal secretsmanager create-secret \
  --name hotel-mlops/db-password \
  --secret-string '{"db_password":"local-dev-password"}'

echo "LocalStack resources initialized."
