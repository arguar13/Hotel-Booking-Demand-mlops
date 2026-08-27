locals {
  ecr_repositories = {
    api       = "${var.project_name}-api"
    dashboard = "${var.project_name}-dashboard"
    # Ships boto3/psycopg2 baked in (Dockerfile.mlflow) - the same image
    # docker-compose.yml uses locally, promoted unchanged to production.
    mlflow = "${var.project_name}-mlflow"
  }
}

resource "aws_ecr_repository" "this" {
  for_each             = local.ecr_repositories
  name                 = each.value
  image_tag_mutability = "MUTABLE"
  force_delete         = true
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
  description = "URLs de los repositorios ECR (api, dashboard, mlflow)"
}
