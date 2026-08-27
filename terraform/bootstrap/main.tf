# One-time bootstrap: the IAM user that every other `terraform` command in this
# repository runs as.
#
# WHY THIS IS A SEPARATE CONFIGURATION, WITH ITS OWN LOCAL STATE
#
# It used to live in ../ alongside the infrastructure, and that is a trap: a
# `terraform destroy` there deletes this user partway through its own run, and
# every remaining API call then fails with InvalidClientTokenId. The teardown
# strands EKS, RDS and the VPC - the expensive things - half-deleted, and
# recovering needs the root key again plus a manual state-lock cleanup.
# An identity must not live in the state it is being used to destroy.
#
# So: apply this once, by itself, with the account root key. Configure the
# resulting credentials as an AWS CLI profile. Everything in ../ then runs as
# that profile, and destroying ../ cannot touch the identity doing the
# destroying. Tear this down last, on its own, with the root key again.
#
#   cd terraform/bootstrap
#   AWS_ACCESS_KEY_ID=<root> AWS_SECRET_ACCESS_KEY=<root> terraform init
#   AWS_ACCESS_KEY_ID=<root> AWS_SECRET_ACCESS_KEY=<root> terraform apply
#
#   AKID=$(terraform output -raw automation_user_access_key_id)
#   SECRET=$(terraform output -raw automation_user_secret_access_key)
#   aws configure set aws_access_key_id     "$AKID"   --profile hotel-mlops
#   aws configure set aws_secret_access_key "$SECRET" --profile hotel-mlops
#   aws configure set region                us-east-1 --profile hotel-mlops
#
# Local state on purpose: the remote backend in ../provider.tf lives in an S3
# bucket that this user is meant to be able to reach, and putting the
# credentials' own state behind a bucket they unlock is the same circular
# dependency one level up. terraform.tfstate here is gitignored; it holds a
# secret access key, so treat it accordingly, or delete it once the profile is
# configured (the user can always be re-imported or recreated).
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "Region de AWS (debe coincidir con la de ../variables.tf)"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre base del proyecto (debe coincidir con el de ../variables.tf)"
  type        = string
  default     = "hotel-mlops"
}

# AdministratorAccess, not a hand-scoped policy: this identity is the one that
# *creates* IAM roles/policies/the OIDC provider, the VPC, EKS, RDS, S3 and ECR,
# and runs helm_release against the cluster it just created. Enumerating a
# policy that covers all of that with no gap that aborts a mid-apply is not
# worth it for a bootstrap identity meant to be short-lived. Day-to-day CI does
# not use it at all: that goes through GitLabCIRole (../iam.tf) via OIDC, with
# no static credentials anywhere.
resource "aws_iam_user" "automation" {
  name = "${var.project_name}-terraform-automation"

  tags = {
    Project = var.project_name
    Purpose = "One-time IaC bootstrap identity - replaces use of the account root key"
  }
}

resource "aws_iam_user_policy_attachment" "automation_admin" {
  user       = aws_iam_user.automation.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_iam_access_key" "automation" {
  user = aws_iam_user.automation.name
}

output "automation_user_access_key_id" {
  value       = aws_iam_access_key.automation.id
  description = "Access key ID - configure as the 'hotel-mlops' AWS CLI profile"
}

output "automation_user_secret_access_key" {
  value       = aws_iam_access_key.automation.secret
  description = "Secret access key for the Terraform automation user"
  sensitive   = true
}
