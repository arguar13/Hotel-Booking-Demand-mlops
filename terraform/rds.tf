# La contrasena se genera aca con `random_password` y el operador la copia a
# mano a un Secret de Kubernetes plano (ver kubernetes/base/mlops-config.env
# y el README, seccion "Desplegar"). Es menos automatico que materializar el
# secreto desde AWS, pero evita mantener piezas de infraestructura extra
# para un proyecto de este tamano. Con mas de un operador y rotacion de
# credenciales, automatizar esa materializacion vuelve a valer la pena.
resource "random_password" "db_password" {
  length  = 24
  special = false # facilita copiar/pegar el valor al crear el Secret de Kubernetes
}

resource "aws_db_instance" "mlflow_db" {
  identifier        = "${var.project_name}-db"
  engine            = "postgres"
  engine_version    = "18.3"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  db_name           = "postgres"
  username          = var.db_username
  password          = random_password.db_password.result

  # Instancia unica, sin Multi-AZ ni replicas de lectura: para un proyecto
  # de aprendizaje, la disponibilidad extra no justifica el costo ni la
  # complejidad de operar un failover.
  multi_az = false

  # Utiliza el subnet group creado en vpc.tf
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]

  skip_final_snapshot = true
  publicly_accessible = false

  # Cifrado en reposo con la clave gestionada por AWS para RDS (aws/rds).
  # No se puede activar sobre una instancia existente: cambiarlo fuerza el
  # reemplazo de la instancia.
  storage_encrypted = true

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

output "rds_endpoint" {
  value       = aws_db_instance.mlflow_db.endpoint
  description = "Endpoint de RDS para MLflow (usar en kubernetes/overlays/production/mlops-config.env)"
}

output "db_password" {
  value       = random_password.db_password.result
  description = "Contrasena generada para RDS. Copiala al crear el Secret de Kubernetes (ver README, seccion Desplegar): `terraform output -raw db_password`"
  sensitive   = true
}
