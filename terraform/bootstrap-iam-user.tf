# One-time bootstrap: an IAM *user* with static credentials, applied once
# using the AWS account's root access key (the only credential that can
# exist before this file is applied - classic chicken-and-egg). Every other
# resource in this Terraform config, and every subsequent `terraform apply`,
# uses this user's credentials instead - the root access key is deleted
# right after this resource is created (see GUIA_EJECUCION_PROYECTO 610.docx,
# section "Bootstrap del usuario IAM de automatizacion").
#
# AdministratorAccess, not a hand-scoped policy: this identity is the one
# that *creates* IAM roles/policies/the OIDC provider (iam.tf), the VPC,
# EKS, RDS, S3, ECR, and runs `helm_release` against the cluster it just
# created - enumerating a policy document that covers all of that exactly,
# with zero gaps that abort a mid-apply, is not worth it for a bootstrap
# identity that is meant to be short-lived. Once the stack is stable, this
# user's access key can be deactivated; day-to-day CI already runs through
# `GitLabCIRole` (iam.tf) via OIDC, with no static credentials at all.
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
  description = "Access key ID for the Terraform automation user - configure as the 'hotel-mlops' AWS CLI profile"
}

output "automation_user_secret_access_key" {
  value       = aws_iam_access_key.automation.secret
  description = "Secret access key for the Terraform automation user"
  sensitive   = true
}
