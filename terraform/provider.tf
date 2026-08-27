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
  #   aws s3api put-bucket-versioning --bucket <project>-tfstate-<account-id> \
  #     --versioning-configuration Status=Enabled
  #
  # then: terraform init \
  #   -backend-config="bucket=<project>-tfstate-<account-id>" \
  #   -backend-config="key=hotel-mlops/terraform.tfstate" \
  #   -backend-config="region=us-east-1" \
  #   -backend-config="use_lockfile=true"
  #
  # State locking uses S3's own conditional writes (`use_lockfile`), not a
  # DynamoDB table: `dynamodb_table` is deprecated as of Terraform 1.11 and
  # S3-native locking needs no second resource to provision, pay for, or
  # keep in sync with the bucket. Bucket versioning above is what makes a
  # corrupted or truncated state recoverable.
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
