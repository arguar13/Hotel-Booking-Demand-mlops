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
    Por defecto abierto, porque el operador necesita kubectl desde una IP
    domestica dinamica y CI NO usa el cluster (el despliegue es GitOps: ArgoCD
    hace pull desde dentro). En un entorno real, restringelo a la IP de salida
    de tu oficina/VPN, o pon cluster_endpoint_public_access = false en eks.tf y
    accede via bastion. Ver AWS-0040/AWS-0041 en .trivyignore.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "db_username" {
  description = "Usuario administrador de RDS PostgreSQL"
  type        = string
  default     = "mlops_user"
}

# ------------------------------------------------------------------
# MSK (terraform/msk.tf) - feed de eventos de prediccion para
# core_ml/src/monitoring/stream_consumer.py
# ------------------------------------------------------------------
variable "msk_kafka_version" {
  description = "Version de Kafka del cluster MSK"
  type        = string
  default     = "3.9.0"
}

variable "msk_instance_type" {
  description = "Tipo de instancia de broker MSK. t3.small alcanza para el volumen de telemetria de este proyecto (early-warning, no un bus de eventos de produccion)"
  type        = string
  default     = "kafka.t3.small"
}

variable "msk_ebs_volume_size" {
  description = "GiB de almacenamiento EBS por broker MSK"
  type        = number
  default     = 100
}
