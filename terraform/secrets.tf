# ============================================================
# Token de GitLab Pipeline Trigger, consumido por
# core_ml/src/monitoring/mitigation.py (corre dentro de jobs-sa: el
# CronJob de drift-monitor y el Deployment de stream-consumer) para
# disparar un reentrenamiento automatico cuando un ALERT sostenido lo
# justifica.
#
# A diferencia de la API key de /predict (api/main.py), este secreto NO
# puede generarse con random_password: es un token que GitLab emite
# (Settings > CI/CD > Pipeline triggers de este proyecto), no algo que
# Terraform pueda inventar. Se declara sin default y sensitive = true;
# se provee en el momento de terraform apply (-var o TF_VAR_) y nunca se
# commitea.
#
# No hace falta tocar ninguna IAM policy para que jobs-sa pueda leerlo:
# external_secrets_read (terraform/iam.tf) ya concede
# secretsmanager:GetSecretValue sobre todo "${var.project_name}/*", y ese
# es el unico rol IRSA que efectivamente llama a Secrets Manager - los
# pods de la aplicacion solo leen el Secret de Kubernetes que ESO ya
# materializo, vía el ExternalSecret de abajo.
# ============================================================

variable "gitlab_pipeline_trigger_token" {
  description = "Token de Pipeline Trigger de GitLab (Settings > CI/CD > Pipeline triggers). Usado por mitigation.py para disparar retrains automaticos."
  type        = string
  sensitive   = true
}

resource "aws_secretsmanager_secret" "gitlab_pipeline_trigger_token" {
  name        = "${var.project_name}/gitlab-pipeline-trigger-token"
  description = "Token de Pipeline Trigger de GitLab, consumido por el ExternalSecret 'mlops-secrets' (clave GITLAB_TRIGGER_TOKEN)"

  recovery_window_in_days = var.environment == "dev" ? 0 : 30
}

resource "aws_secretsmanager_secret_version" "gitlab_pipeline_trigger_token" {
  secret_id     = aws_secretsmanager_secret.gitlab_pipeline_trigger_token.id
  secret_string = jsonencode({ gitlab_trigger_token = var.gitlab_pipeline_trigger_token })
}
