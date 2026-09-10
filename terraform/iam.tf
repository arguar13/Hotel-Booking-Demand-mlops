# El cluster EKS y el node group ya reciben sus propios roles IAM del modulo
# terraform-aws-modules/eks/aws (eks.tf) - no hace falta declararlos a mano.
# Lo que este archivo si declara:
#
#   1. La politica de S3 que la API, MLflow y los jobs batch necesitan para
#      leer/escribir el bucket de artefactos - adjuntada directamente al rol
#      IAM del node group (ver eks.tf's iam_role_additional_policies), en vez
#      de un rol IRSA por ServiceAccount. Es menos granular (cualquier pod
#      del nodo hereda el acceso), pero evita levantar el proveedor OIDC y
#      mantener un rol por servicio para un unico node group y un unico
#      bucket. Con varios equipos compartiendo el cluster, un rol IRSA por
#      ServiceAccount pasa a valer la pena.
#   2. El rol que CI (GitLab) asume via OIDC para construir/publicar
#      imagenes, leer/escribir el bucket de DVC y desplegar con kubectl.

data "aws_iam_policy_document" "artifacts_bucket_rw" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.mlflow_dvc_artifacts.arn,
      "${aws_s3_bucket.mlflow_dvc_artifacts.arn}/*",
    ]
  }
}

resource "aws_iam_policy" "artifacts_bucket_rw" {
  name   = "${var.project_name}-artifacts-bucket-rw"
  policy = data.aws_iam_policy_document.artifacts_bucket_rw.json
}

# ============================================================
# OIDC federation para GitLab CI/CD - permite a .gitlab-ci.yml asumir un rol
# de AWS sin credenciales estaticas (access key/secret) guardadas en GitLab.
# GitLab expone su propio emisor OIDC vía `id_tokens:` (ver .gitlab-ci.yml);
# aca se lee el proveedor (normalmente ya registrado por otro proyecto de la
# misma cuenta) y se restringe que solo este proyecto/rama pueda asumir el rol.
# ============================================================
variable "gitlab_oidc_issuer_url" {
  description = "URL del emisor OIDC de GitLab (https://gitlab.com para SaaS, o la URL de tu instancia self-managed)"
  type        = string
  default     = "https://gitlab.com"
}

variable "gitlab_project_path" {
  description = "Ruta del proyecto en GitLab, ej. \"mi-grupo/hotel-mlops\" - restringe qué pipeline puede asumir GitLabCIRole"
  type        = string
  default     = "CHANGE_ME/hotel-mlops"
}

# Un proveedor OIDC IAM es un singleton por URL de emisor *para toda la
# cuenta de AWS*, no por estado de Terraform - se lee como data en vez de
# gestionarse como resource para adoptar el que ya exista sin crear ni
# destruir un recurso de toda la cuenta del que puede depender otro proyecto.
data "aws_iam_openid_connect_provider" "gitlab" {
  url = var.gitlab_oidc_issuer_url
}

resource "aws_iam_role" "gitlab_ci_role" {
  name = "${var.project_name}-gitlab-ci"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = data.aws_iam_openid_connect_provider.gitlab.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.gitlab_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "${replace(var.gitlab_oidc_issuer_url, "https://", "")}:sub" = "project_path:${var.gitlab_project_path}:ref_type:branch:ref:*"
          }
        }
      }
    ]
  })
}

# CI necesita: (1) publicar imagenes en ECR, (2) leer/escribir el bucket de
# artefactos S3 (DVC push/pull, lectura del dataset), y (3) desplegar con
# `kubectl apply -k` en el job `deploy` de .gitlab-ci.yml.
resource "aws_iam_role_policy_attachment" "gitlab_ci_ecr_push" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_s3_dvc" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = aws_iam_policy.artifacts_bucket_rw.arn
}

# Da a gitlab_ci_role permiso de kubectl sobre el cluster (EKS Access Entry).
# AmazonEKSAdminPolicy es intencionalmente amplia para este alcance: en un
# entorno real, restringila a la namespace hotel-mlops con una politica de
# acceso mas angosta.
resource "aws_eks_access_entry" "gitlab_ci" {
  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.gitlab_ci_role.arn
}

resource "aws_eks_access_policy_association" "gitlab_ci_admin" {
  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.gitlab_ci_role.arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSAdminPolicy"

  access_scope {
    type = "cluster"
  }
}

output "gitlab_ci_role_arn" {
  value       = aws_iam_role.gitlab_ci_role.arn
  description = "ARN to set as AWS_ROLE_ARN in .gitlab-ci.yml"
}
