module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.5.0"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  # Subnets exclusivas para la Base de Datos (RDS)
  database_subnets             = ["10.0.201.0/24", "10.0.202.0/24"]
  create_database_subnet_group = true

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Security Group para los nodos de Kubernetes (EKS)
resource "aws_security_group" "eks_nodes_sg" {
  name        = "${var.project_name}-eks-nodes-sg"
  description = "Security group para nodos worker de EKS"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Sin reglas de ingreso, este grupo dejaba pasar todo el trafico saliente de
# los nodos pero ninguno entrante - ademas del cluster primary security group
# que el modulo EKS adjunta por su cuenta, el control plane necesita alcanzar
# el kubelet de cada nodo (puerto 10250) para servir `kubectl logs`/`exec`, y
# los nodos necesitan hablarse entre si para el trafico de pod a pod entre
# distintas instancias EC2 (el CNI de VPC no lo cubre solo con el SG del
# cluster). Sin esto, `kubectl get pods` funciona (el kubelet reporta su
# propio estado hacia afuera) pero `kubectl logs`/`exec` fallan con "TLS
# handshake timeout" - un sintoma de red, no un problema de la aplicacion.
# Reglas minimas recomendadas por AWS para el SG de nodos EKS:
# https://docs.aws.amazon.com/eks/latest/userguide/sec-group-reqs.html
resource "aws_security_group_rule" "eks_nodes_from_cluster_kubelet" {
  description              = "Kubelet API (logs/exec/metrics) desde el control plane de EKS"
  type                     = "ingress"
  from_port                = 10250
  to_port                  = 10250
  protocol                 = "tcp"
  security_group_id        = aws_security_group.eks_nodes_sg.id
  source_security_group_id = module.eks.cluster_security_group_id
}

resource "aws_security_group_rule" "eks_nodes_from_cluster_webhooks" {
  description              = "Puertos efimeros para webhooks/API servers de extension (ej. metrics-server) desde el control plane"
  type                     = "ingress"
  from_port                = 1025
  to_port                  = 65535
  protocol                 = "tcp"
  security_group_id        = aws_security_group.eks_nodes_sg.id
  source_security_group_id = module.eks.cluster_security_group_id
}

resource "aws_security_group_rule" "eks_nodes_self" {
  description       = "Trafico de pod a pod entre nodos distintos (CNI)"
  type              = "ingress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  security_group_id = aws_security_group.eks_nodes_sg.id
  self              = true
}

# Security Group para RDS asegurando aislamiento
resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Permitir trafico PostgreSQL solo desde nodos EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "Trafico interno desde nodos EKS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes_sg.id] # Vinculo explicito
  }
}
