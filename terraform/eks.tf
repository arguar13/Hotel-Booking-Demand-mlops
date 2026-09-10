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
  # reemplazo al viejo ConfigMap aws-auth, y sin una entrada explicita
  # `kubectl` (a mano o desde el job `deploy` de CI) falla con "Unauthorized".
  # Esto crea la access entry del creador con la politica AmazonEKSClusterAdminPolicy.
  enable_cluster_creator_admin_permissions = true

  # Sin IRSA (IAM Roles for Service Accounts): en vez de un rol IAM distinto
  # por ServiceAccount, el unico node group de abajo lleva la politica de S3
  # que la API y MLflow necesitan (ver iam_role_additional_policies). Es
  # menos granular - cualquier pod en el nodo hereda ese acceso a S3 en vez
  # de solo el ServiceAccount que lo declara - pero para un solo node group
  # y un solo bucket, evita levantar el proveedor OIDC y un rol por servicio.
  enable_irsa = false

  eks_managed_node_groups = {
    # Un unico node group: para el volumen de este proyecto no hace falta
    # separar pools por tipo de carga (serving vs. batch); todo corre sobre
    # las mismas instancias t3.medium.
    general = {
      desired_size = 2
      min_size     = 1
      max_size     = 3

      instance_types         = ["t3.medium"]
      vpc_security_group_ids = [aws_security_group.eks_nodes_sg.id]

      # Da a cualquier pod del cluster (api, mlflow, jobs) acceso de
      # lectura/escritura al bucket de artefactos S3, sin un rol IAM por
      # ServiceAccount - ver iam.tf.
      iam_role_additional_policies = {
        s3_artifacts = aws_iam_policy.artifacts_bucket_rw.arn
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}
