# Bootstraps the External Secrets Operator (ESO) into the cluster via Helm.
# ESO is what lets kubernetes/base/external-secret-db.yaml (committed to
# Git, no secret material inside it) materialize into a real Kubernetes
# Secret at runtime by reading AWS Secrets Manager directly - the
# declarative replacement for the old `sed -i s/__DB_PASSWORD_PLACEHOLDER__/`
# hack in .gitlab-ci.yml's deploy step.
resource "kubernetes_namespace" "external_secrets" {
  metadata {
    name = "external-secrets"
  }
}

resource "helm_release" "external_secrets" {
  name       = "external-secrets"
  repository = "https://charts.external-secrets.io"
  chart      = "external-secrets"
  version    = "0.10.4"
  namespace  = kubernetes_namespace.external_secrets.metadata[0].name

  set {
    name  = "serviceAccount.name"
    value = "external-secrets-sa"
  }

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.external_secrets_irsa_role.iam_role_arn
  }

  depends_on = [module.eks]
}
