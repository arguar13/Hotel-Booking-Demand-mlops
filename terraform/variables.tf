variable "aws_region" {
  description = "Región de AWS para el despliegue"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre base para los recursos del proyecto"
  type        = string
  default     = "hotel-mlops"
}

variable "environment" {
  description = "Entorno de despliegue (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "Rango CIDR de la VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = <<-EOT
    Rangos CIDR que pueden alcanzar el endpoint publico de la API de EKS.
    Por defecto abierto, porque tanto el operador (kubectl desde una IP
    domestica dinamica) como el runner de CI (el job `deploy` de
    .gitlab-ci.yml corre `kubectl apply -k` directamente) necesitan llegar
    al cluster. En un entorno real, restringelo a las IPs de salida
    conocidas (oficina/VPN, runner de CI), o pon
    cluster_endpoint_public_access = false en eks.tf y accede via bastion.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "db_username" {
  description = "Usuario administrador de RDS PostgreSQL"
  type        = string
  default     = "mlops_user"
}
