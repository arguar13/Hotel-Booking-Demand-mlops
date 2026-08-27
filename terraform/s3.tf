resource "aws_s3_bucket" "mlflow_dvc_artifacts" {
  bucket = "${var.project_name}-artifacts-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Este bucket guarda modelos entrenados y datasets versionados con DVC: nada
# de eso debe ser alcanzable publicamente bajo ninguna circunstancia. Los
# cuatro flags se declaran de forma explicita para que un ACL o una policy
# publica puesta por error (por una herramienta, o a mano) no tenga efecto.
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.mlflow_dvc_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Cifrado en reposo con SSE-S3 (AES-256, claves gestionadas por AWS). Ver la
# entrada AWS-0132 en .trivyignore para por que no se usa una CMK aqui.
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.mlflow_dvc_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
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
