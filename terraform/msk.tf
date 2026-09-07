# Amazon MSK: near-real-time feed for core_ml/src/monitoring/stream_consumer.py.
#
# api/main.py has published well-formed prediction events since before this
# file existed (`_publish_prediction_event`, gated on KAFKA_BOOTSTRAP_SERVERS)
# but no .tf ever provisioned a broker for it to reach, so the publish path
# was permanently a no-op in production. See api/monitoring.py's module
# docstring for why that was a deliberate decision at the time, and this
# file's own header comment in stream_consumer.py for why it no longer holds.
#
# Deliberately small: 2 brokers (one per private AZ, matching module.vpc),
# kafka.t3.small - this carries prediction-event telemetry for drift
# early-warning, not a production event bus. Bump broker_node_group_info if
# traffic ever justifies it.
resource "aws_security_group" "msk_sg" {
  name        = "${var.project_name}-msk-sg"
  description = "Permitir trafico Kafka (TLS, 9094) exclusivamente desde EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    # Puerto 9094 = listener TLS de MSK. Con client_broker = "TLS" el 9092
    # (plaintext) ni siquiera esta abierto en el broker.
    description     = "Kafka broker (TLS)"
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes_sg.id]
  }
}

resource "aws_msk_cluster" "kafka" {
  cluster_name           = "${var.project_name}-kafka"
  kafka_version          = var.msk_kafka_version
  number_of_broker_nodes = length(module.vpc.private_subnets)

  broker_node_group_info {
    instance_type   = var.msk_instance_type
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.msk_sg.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.msk_ebs_volume_size
      }
    }
  }

  encryption_info {
    # Sin CMK propia a proposito: este proyecto ya tomo esa decision para S3
    # (ver .trivyignore AWS-0132 - sin requisito regulatorio ni necesidad de
    # rotacion on-demand, una CMK solo suma costo por request y una key
    # policy que mantener). MSK cifra en reposo con la clave gestionada por
    # AWS (aws/kafka) cuando encryption_at_rest_kms_key_arn se omite, lo que
    # mantiene la misma postura en todo el proyecto en vez de introducir una
    # CMK solo para este recurso.
    encryption_in_transit {
      # PLAINTEXT dejaria cada evento de prediccion (features + prediccion +
      # confianza) en claro para cualquiera con acceso de red a la VPC. TLS
      # obligatorio: el cliente kafka-python de api/main.py se conecta con
      # security_protocol="SSL" - ver KAFKA_SECURITY_PROTOCOL en
      # kubernetes/base/*.env. MSK usa certificados de Amazon Trust Services,
      # ya presentes en el almacen de CAs del sistema.
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.project_name}"
  retention_in_days = 14
}

# Con client_broker = "TLS" el cluster NO expone el listener 9092 en claro:
# el output bootstrap_brokers (plaintext) queda vacio. El endpoint correcto
# es el TLS (puerto 9094), inyectado como KAFKA_BOOTSTRAP_SERVERS via
# kubernetes/base/mlops-config.env (overlay de produccion).
output "msk_bootstrap_brokers" {
  description = "Usar como KAFKA_BOOTSTRAP_SERVERS (listener TLS, puerto 9094)"
  value       = aws_msk_cluster.kafka.bootstrap_brokers_tls
}
