resource "aws_s3_bucket" "mlflow_dvc_artifacts" {
  bucket = "${var.project_name}-artifacts-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Versionado de S3 obligatorio para MLOps
resource "aws_s3_bucket_versioning" "artifacts_versioning" {
  bucket = aws_s3_bucket.mlflow_dvc_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Regla de ciclo de vida para transicionar modelos antiguos a almacenamiento frio
resource "aws_s3_bucket_lifecycle_configuration" "artifacts_lifecycle" {
  bucket = aws_s3_bucket.mlflow_dvc_artifacts.id

  rule {
    id     = "archive-old-artifacts"
    status = "Enabled"

    # Empty filter = "every object in the bucket". Required explicitly:
    # a rule with neither `filter` nor `prefix` is a provider warning today
    # and an error in a future AWS provider version.
    filter {}

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }
}

output "s3_bucket_name" {
  value = aws_s3_bucket.mlflow_dvc_artifacts.id
}
