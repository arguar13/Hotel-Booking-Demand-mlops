# Lectura del secreto en Secrets Manager
data "aws_secretsmanager_secret" "db_password" {
  # Asegúrate de que este nombre coincida exactamente con el que pusiste en el Paso 1
  name = "hotel-mlops/db-password"
}

data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = data.aws_secretsmanager_secret.db_password.id
}

# Inyección del secreto en un local
locals {
  db_password = jsondecode(data.aws_secretsmanager_secret_version.db_password.secret_string)["db_password"]
}

resource "aws_db_instance" "mlflow_db" {
  identifier        = "${var.project_name}-db"
  engine            = "postgres"
  engine_version    = "18.3"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  db_name           = "postgres"
  username          = var.db_username

  # Usamos el local generado desde Secrets Manager en lugar de var.db_password
  password = local.db_password

  # Utiliza el subnet group creado en vpc.tf
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]

  skip_final_snapshot = true
  publicly_accessible = false

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

output "rds_endpoint" {
  value       = aws_db_instance.mlflow_db.endpoint
  description = "Endpoint de RDS para MLflow (Inyectar en ConfigMap K8s)"
}
