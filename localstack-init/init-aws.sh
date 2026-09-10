#!/bin/bash
# Runs automatically once LocalStack's edge service is ready
# (LocalStack executes every *.sh file under /etc/localstack/init/ready.d/).
# Creates the S3 buckets MLflow's artifact store and DVC's remote need
# locally, mirroring the one S3 bucket terraform/s3.tf provisions in real AWS.
set -euo pipefail

echo "==> Creating S3 buckets"
awslocal s3 mb s3://mlflow-artifacts
awslocal s3 mb s3://hotel-mlops-dvc-store

echo "LocalStack resources initialized."
