# Scoped to the artifacts bucket only (not a blanket S3FullAccess/
# AmazonS3ReadOnlyAccess managed policy) - both mlflow (artifact store)
# and the api (loading models from that same store) only ever need this
# one bucket.
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

# IRSA (IAM Roles for Service Accounts) para MLflow
module "mlflow_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"

  role_name = "${var.project_name}-mlflow-irsa"

  role_policy_arns = {
    artifacts_bucket_rw = aws_iam_policy.artifacts_bucket_rw.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:mlflow-sa"]
    }
  }

  tags = {
    Environment = var.environment
  }
}

output "mlflow_iam_role_arn" {
  value       = module.mlflow_irsa_role.iam_role_arn
  description = "ARN del rol IAM para el ServiceAccount mlflow-sa"
}

module "api_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"

  role_name = "${var.project_name}-api-irsa"

  role_policy_arns = {
    artifacts_bucket_rw = aws_iam_policy.artifacts_bucket_rw.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:api-sa"]
    }
  }
}

output "api_iam_role_arn" {
  value = module.api_irsa_role.iam_role_arn
}

# IRSA for in-cluster batch workloads (the drift-monitor CronJob today). It
# needs the same artifacts bucket the API and MLflow use - read, to pull the
# reference profile logged with the served model version; write, because the
# drift report it produces is itself an MLflow artifact - but it is a distinct
# workload with a distinct lifecycle, so it gets a distinct role rather than
# borrowing api-sa's. Reusing the serving identity would mean every permission
# a future batch job needs is silently granted to the internet-facing API too.
module "jobs_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"

  role_name = "${var.project_name}-jobs-irsa"

  role_policy_arns = {
    artifacts_bucket_rw = aws_iam_policy.artifacts_bucket_rw.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:jobs-sa"]
    }
  }

  tags = {
    Environment = var.environment
  }
}

output "jobs_iam_role_arn" {
  value       = module.jobs_irsa_role.iam_role_arn
  description = "ARN to annotate on the jobs-sa ServiceAccount (drift-monitor CronJob)"
}

# ============================================================
# IRSA para External Secrets Operator: lee AWS Secrets Manager y
# materializa Secrets de Kubernetes a partir de un ExternalSecret
# declarado en Git (kubernetes/base/external-secret-db.yaml) - el
# valor del secreto nunca vive en el repo, solo la *referencia* a él.
# ============================================================
data "aws_iam_policy_document" "external_secrets_read" {
  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
    ]
    resources = [
      "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/*"
    ]
  }
}

resource "aws_iam_policy" "external_secrets_read" {
  name   = "${var.project_name}-external-secrets-read"
  policy = data.aws_iam_policy_document.external_secrets_read.json
}

module "external_secrets_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"

  role_name = "${var.project_name}-external-secrets-irsa"

  role_policy_arns = {
    secretsmanager_read = aws_iam_policy.external_secrets_read.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["external-secrets:external-secrets-sa"]
    }
  }
}

output "external_secrets_iam_role_arn" {
  value       = module.external_secrets_irsa_role.iam_role_arn
  description = "ARN a anotar en kubernetes/base/external-secrets-serviceaccount.yaml"
}

# ============================================================
# OIDC federation para GitLab CI/CD - reemplaza al viejo
# assume_role_policy roto (confiaba en "ec2.amazonaws.com", que
# nunca puede emitir el token web-identity que .gitlab-ci.yml
# realmente envía). GitLab expone su propio emisor OIDC vía
# `id_tokens:` (ver .gitlab-ci.yml); aquí se registra como
# proveedor y se restringe qué proyecto/rama puede asumir el rol.
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

# An IAM OIDC provider is a singleton per issuer URL *for the whole AWS
# account*, not per Terraform state - a second `resource` block trying to
# create "https://gitlab.com" again fails with EntityAlreadyExists the moment
# any other project in this account has already registered it (verified here:
# it already exists, created by a different project's Terraform, tagged
# accordingly). Reading it as data instead of managing it as a resource means
# this configuration adopts whichever provider is already there - with the
# same client_id_list/thumbprint any GitLab.com issuer produces regardless of
# who created it - without ever creating or destroying an account-wide
# resource a sibling project also depends on.
data "aws_iam_openid_connect_provider" "gitlab" {
  url = var.gitlab_oidc_issuer_url
}

# Named "GitLabCIRole" until this deployment: a bare, unprefixed name that
# collided with an identically-named role from another project in this same
# AWS account (each with its own OIDC trust condition scoped to a different
# gitlab_project_path) - IAM role names are unique per account, not per
# Terraform state. Prefixed with project_name like every other resource here
# to make that collision structurally impossible going forward.
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

# CI solo necesita: (1) publicar imagenes en ECR, (2) leer/escribir el
# bucket de artefactos S3 (DVC push/pull, lectura de datasets), y
# (3) leer el secreto de RDS para el smoke test de entrenamiento.
# Ya NO necesita permisos de EKS: con GitOps, CI nunca llama a
# `kubectl apply` - ArgoCD reconcilia el cluster desde Git de forma
# independiente (ver gitops/argocd/application.yaml).
resource "aws_iam_role_policy_attachment" "gitlab_ci_ecr_push" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

data "aws_iam_policy_document" "gitlab_ci_s3_dvc" {
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

  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
    ]
    resources = [data.aws_secretsmanager_secret.db_password.arn]
  }
}

resource "aws_iam_policy" "gitlab_ci_s3_dvc" {
  name   = "${var.project_name}-gitlab-ci-s3-dvc"
  policy = data.aws_iam_policy_document.gitlab_ci_s3_dvc.json
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_s3_dvc" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = aws_iam_policy.gitlab_ci_s3_dvc.arn
}

output "gitlab_ci_role_arn" {
  value       = aws_iam_role.gitlab_ci_role.arn
  description = "ARN to set as AWS_ROLE_ARN in .gitlab-ci.yml"
}
