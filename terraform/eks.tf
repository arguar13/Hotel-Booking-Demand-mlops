module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-cluster"
  cluster_version = "1.36"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access       = true
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs

  # Sin esto, NADA puede autenticarse contra el cluster recien creado.
  # A partir de la v20 del modulo, el principal IAM que crea el cluster ya no
  # recibe permisos de Kubernetes automaticamente: EKS Access Entries
  # reemplazo al viejo ConfigMap aws-auth, y sin una entrada explicita el
  # propio `terraform apply` falla con "Unauthorized" en cuanto los providers
  # kubernetes/helm intentan crear algo (los namespaces y los Helm releases de
  # ArgoCD y External Secrets, mas abajo en este mismo apply).
  # Esto crea la access entry del creador con la politica AmazonEKSClusterAdminPolicy.
  enable_cluster_creator_admin_permissions = true

  # Habilita OpenID Connect (OIDC) para usar IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  eks_managed_node_groups = {
    general = {
      # Bumped from 2 to 3 (still within the pre-existing max_size) after
      # ip-10-0-2-176 went NotReady (kubelet stopped posting status) mid
      # rollout: its pods stuck in Terminating (can't be force-deleted
      # without direct node/pod-deletion access) left the one healthy node
      # at its hard 17-pod ENI limit for t3.medium, blocking the correctly
      # configured api replica from ever scheduling. Adding real capacity
      # instead of fighting Kubernetes' own node-eviction bookkeeping.
      # Note: the eks module ignores drift on scaling_config.desired_size
      # (an external autoscaler is expected to own it after creation), so
      # this edit alone does not resize anything - the actual scale-up used
      # `aws eks update-nodegroup-config`. This value is kept in sync with
      # that so a future `terraform apply` does not try to fight it back
      # down to 2.
      desired_size = 3
      min_size     = 1
      max_size     = 3

      instance_types         = ["t3.medium"]
      vpc_security_group_ids = [aws_security_group.eks_nodes_sg.id]
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}
