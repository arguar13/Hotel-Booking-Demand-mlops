terraform {
  required_version = ">= 1.5.0"

  # Remote state, not local .tfstate (which is gitignored - it must never
  # be committed, and a local-only file means state is lost the moment a
  # different machine or CI runner runs `terraform apply`). Values are
  # deliberately left out of this block ("partial configuration") since
  # the state bucket must exist *before* this config's own S3 bucket does
  # - bootstrap it once by hand:
  #
  #   aws s3api create-bucket --bucket <project>-tfstate-<account-id> --region us-east-1
  #   aws dynamodb create-table --table-name <project>-tfstate-lock \
  #     --attribute-definitions AttributeName=LockID,AttributeType=S \
  #     --key-schema AttributeName=LockID,KeyType=HASH \
  #     --billing-mode PAY_PER_REQUEST
  #
  # then: terraform init -backend-config="bucket=<project>-tfstate-<account-id>" \
  #   -backend-config="key=hotel-mlops/terraform.tfstate" \
  #   -backend-config="region=us-east-1" \
  #   -backend-config="dynamodb_table=<project>-tfstate-lock"
  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.14"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Autenticacion contra el cluster EKS recien creado, reutilizada por los
# providers kubernetes/helm de abajo (bootstrap de ArgoCD y External
# Secrets - ver argocd.tf y external-secrets.tf).
data "aws_eks_cluster_auth" "this" {
  name = module.eks.cluster_name
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}

# Obtiene la identidad actual (Account ID)
data "aws_caller_identity" "current" {}
