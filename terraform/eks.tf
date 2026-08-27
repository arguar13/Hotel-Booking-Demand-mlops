module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-cluster"
  cluster_version = "1.36"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  # Habilita OpenID Connect (OIDC) para usar IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  eks_managed_node_groups = {
    general = {
      desired_size = 2
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
