locals {
  ecr_repositories = {
    api       = "${var.project_name}-api"
    dashboard = "${var.project_name}-dashboard"
    # Batch workloads: core_ml's code packaged to run inside the cluster
    # (Dockerfile.jobs) - the drift-check CronJob and manual training/promotion
    # runs all use it, with a different `command:`.
    jobs = "${var.project_name}-jobs"
    # MLflow itself is NOT built here: kubernetes/base/mlflow.yaml runs the
    # official ghcr.io/mlflow/mlflow image directly, so there is no image for
    # this project's own CI to build or push for it.
  }
}

resource "aws_ecr_repository" "this" {
  for_each = local.ecr_repositories
  name     = each.value
  # Tags mutables: permite volver a publicar ":latest". En un entorno real,
  # IMMUTABLE evita sobrescribir un tag ya desplegado por error.
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  # Escaneo basico de vulnerabilidades en cada push (gratis, integrado en
  # ECR) - no reemplaza un scanner dedicado, pero es mejor que nada sin
  # agregar una herramienta mas al pipeline.
  image_scanning_configuration {
    scan_on_push = true
  }
}

# Politica de retencion para ahorrar costos (mantiene solo las ultimas 30 imagenes)
resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Retener solo las ultimas 30 imagenes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 30
      }
      action = {
        type = "expire"
      }
    }]
  })
}

output "ecr_repository_urls" {
  value       = { for k, v in aws_ecr_repository.this : k => v.repository_url }
  description = "URLs de los repositorios ECR (api, dashboard, jobs)"
}
