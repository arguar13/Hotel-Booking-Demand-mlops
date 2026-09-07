locals {
  ecr_repositories = {
    api       = "${var.project_name}-api"
    dashboard = "${var.project_name}-dashboard"
    # Ships boto3/psycopg2 baked in (Dockerfile.mlflow) - the same image
    # docker-compose.yml uses locally, promoted unchanged to production.
    mlflow = "${var.project_name}-mlflow"
    # Batch workloads: core_ml's code packaged to run inside the cluster
    # (Dockerfile.jobs). The drift-monitor CronJob runs from it today; a
    # scheduled retrain would reuse the same image with a different command.
    # Until this image existed, none of the three carried the ML code at all,
    # so nothing on the training/monitoring side could run outside a laptop.
    jobs = "${var.project_name}-jobs"
  }
}

resource "aws_ecr_repository" "this" {
  for_each = local.ecr_repositories
  name     = each.value
  # Ver la entrada AWS-0031 en .trivyignore para por que los tags son mutables.
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  # Escaneo de vulnerabilidades en cada push. Cierra el circulo con el job
  # trivy-scan de CI: aquel analiza el codigo y los lockfiles antes de
  # construir; esto analiza la imagen ya construida, incluidos los paquetes
  # del sistema operativo base que el lockfile no cubre.
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
  description = "URLs de los repositorios ECR (api, dashboard, mlflow, jobs)"
}
