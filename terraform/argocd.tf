# Bootstraps ArgoCD into the cluster via Helm. This is the one and only
# piece of "day-1" cluster setup Terraform does for GitOps - everything
# *application-level* after this point (which image tag is running, which
# ConfigMap values are set) is owned by ArgoCD reconciling
# gitops/argocd/application.yaml against kubernetes/overlays/production,
# not by Terraform or by CI running `kubectl apply`.
resource "kubernetes_namespace" "argocd" {
  metadata {
    name = "argocd"
  }
}

resource "helm_release" "argocd" {
  name       = "argocd"
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-cd"
  version    = "7.7.11"
  namespace  = kubernetes_namespace.argocd.metadata[0].name

  # El self-heal automatico (si alguien edita el cluster a mano via
  # kubectl edit, etc., ArgoCD revierte al estado declarado en Git en el
  # siguiente ciclo de sync) se configura en la propia Application, ver
  # gitops/argocd/application.yaml.
  set {
    name  = "configs.params.application\\.namespaces"
    value = "default"
  }

  depends_on = [module.eks]
}

# NOTE on bootstrapping the root Application (deliberately *not* a
# `kubernetes_manifest` resource here): the Application CRD only exists
# once the argocd Helm release above has actually installed it, so
# Terraform can't reliably plan a kubernetes_manifest for it in the same
# apply (classic CRD chicken-and-egg problem). This is a one-time,
# one-line bootstrap step instead - see the "GitOps Bootstrap" section of
# the README:
#
#   kubectl apply -f gitops/argocd/application.yaml
#
# After that single manual step, ArgoCD owns itself: every further change
# to kubernetes/overlays/production/ (image tags, config) is picked up by
# ArgoCD's own sync loop, no Terraform or kubectl involved.
