*Leer este documento en otros idiomas: [English](README.md)*

# Pipeline MLOps de Extremo a Extremo: Predicción del Segmento de Mercado de Reservas Hoteleras

Un pipeline de Machine Learning Operations (MLOps) de extremo a extremo diseñado para predecir segmentos de mercado de reservas hoteleras. Este proyecto cubre el ciclo de vida completo de una aplicación de aprendizaje automático: desde el versionado de datos y la optimización de hiperparámetros hasta la contenerización, la orquestación con Kubernetes, la integración continua y el despliegue de modelos en tiempo real.

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Problema de Negocio](#problema-de-negocio)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Stack Tecnológico](#stack-tecnológico)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos Previos](#requisitos-previos)
- [Configuración y Ejecución Local](#configuración-y-ejecución-local)
  - [1. Clonar el Repositorio y Configurar el Entorno](#1-clonar-el-repositorio-y-configurar-el-entorno)
  - [2. Configuración de LocalStack y Descarga de Datos con DVC](#2-configuración-de-localstack-y-descarga-de-datos-con-dvc)
  - [3. Entrenamiento y Seguimiento del Modelo](#3-entrenamiento-y-seguimiento-del-modelo)
  - [4. Contenerización y Despliegue en Kubernetes](#4-contenerización-y-despliegue-en-kubernetes)
  - [5. Dashboard Interactivo con Streamlit](#5-dashboard-interactivo-con-streamlit)
- [Integración Continua (CI/CD)](#integración-continua-cicd)
- [Pruebas](#pruebas)

---

## Descripción General

Este repositorio demuestra un framework MLOps listo para producción que automatiza el entrenamiento de modelos, el seguimiento de experimentos, el control de versiones y el despliegue contenerizado. El objetivo es proporcionar un servicio de aprendizaje automático reproducible, probado y escalable utilizando simulaciones de servicios en la nube locales para garantizar un costo de infraestructura en la nube igual a cero durante el desarrollo.

### Capacidades Principales

- Versionado reproducible de datos y modelos utilizando DVC respaldado por LocalStack S3.
- Búsqueda automatizada de hiperparámetros con Optuna y seguimiento de experimentos con MLflow.
- Implementación de una API REST utilizando FastAPI con validación completa de solicitudes mediante Pydantic.
- Despliegue contenerizado utilizando Docker y Kubernetes (Minikube).
- Integración Continua mediante GitHub Actions ejecutando pruebas unitarias automatizadas y verificaciones de construcción de Docker.
- Interfaz de usuario para inferencia de modelos impulsada por Streamlit.

---

## Problema de Negocio

En la industria hotelera, comprender las fuentes de reserva y los segmentos de mercado (por ejemplo, Directo, Corporativo, Agencias de Viajes Online) es fundamental para optimizar la gestión de ingresos, la asignación de recursos y las estrategias de marketing.

Este pipeline procesa características históricas de reservas hoteleras, como el tiempo de anticipación, la duración de la estancia, los atributos del cliente y los tipos de habitación, para clasificar y predecir el segmento de mercado de nuevas reservas en tiempo real.

---

## Arquitectura del Sistema

El siguiente diagrama ilustra el flujo de datos, seguimiento, contenerización y despliegue a través del pipeline:

```text
                                 +-----------------------------------+
                                 |       Repositorio Git             |
                                 +-----------------+-----------------+
                                                   |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
       +-------------------+                             +-------------------+
       |   Punteros DVC    |                             | GitHub Actions    |
       |                   |                             |     (CI/CD)       |
       +---------+---------+                             +---------+---------+
                 |                                                 |
                 v                                                 v
       +-------------------+                             +-------------------+
       |   LocalStack S3   |                             | Pytest y Docker   |
       |  (s3://dvc-store) |                             | Verificación Build|
       +-------------------+                             +-------------------+

                         +-----------------------------------+
                         |     Pipeline de Entrenamiento     |
                         | (Optuna + Seguimiento MLflow)     |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |          LocalStack S3            |
                         |      (Artefactos de MLflow)       |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |           Aplicación FastAPI      |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |       Contenedor Docker           |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |     Clúster de Kubernetes         |
                         |          (Minikube)               |
                         +-----------------+-----------------+
                                           ^
                                           |
                         +-----------------+-----------------+
                         |          Interfaz Streamlit       |
                         +-----------------------------------+
```

---

## Stack Tecnológico

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje y Núcleo** | Python 3.12, Pandas, Scikit-learn, Imbalanced-learn |
| **Optimización y Seguimiento** | Optuna, MLflow |
| **Almacenamiento y Versionado** | DVC (Data Version Control), LocalStack (Emulación de AWS S3), Git |
| **API y Validación** | FastAPI, Pydantic, Uvicorn |
| **Interfaz de Usuario** | Streamlit |
| **Contenerización y Orquestación** | Docker, Kubernetes (Minikube), Kubectl |
| **CI/CD y Pruebas** | GitHub Actions, Pytest, HTTPX |

---

## Estructura del Repositorio

```text
hotel-booking-mlops/
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── .git/
├── .gitignore
├── .dockerignore
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── config/
│   └── config.yaml
├── dashboard/
│   └── app.py
├── data/
│   └── hotel_bookings.csv
├── kubernetes/
│   ├── 01-config-secrets.yaml
│   ├── 04-mlflow.yaml
│   ├── 05-api.yaml
│   └── 06-dashboard.yaml
├── models/
│   └── (vacío)
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_processing.py
│   └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── Dockerfile.dashboard
├── README.md
├── README_es.md
└── requirements.txt
```

---

## Requisitos Previos

Asegúrate de tener instaladas las siguientes herramientas en tu sistema local:

- **Python 3.12+**
- **Git**
- **Docker Desktop**
- **Minikube**
- **Kubectl**
- **AWS CLI** (para interactuar con LocalStack)

---

## Configuración y Ejecución Local

### 1. Clonar el Repositorio y Configurar el Entorno

Clona el repositorio y configura un entorno virtual de Python:

```bash
git clone https://github.com/arguar13/booking-hotels-mlops.git
cd booking-hotels-mlops

python -m venv ml_env

# Windows (Git Bash)
source ml_env/Scripts/activate

# Linux/macOS
# source ml_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuración de LocalStack y Descarga de Datos con DVC

Inicia LocalStack y crea los buckets S3 necesarios para el almacenamiento de artefactos y el versionado de datos:

```bash
# Asegúrate de que el contenedor de LocalStack esté ejecutándose en el puerto 4566
docker run -d -p 4566:4566 -p 4571:4571 localstack/localstack

# Crear buckets S3
aws --endpoint-url=http://localhost:4566 s3 mb s3://mlflow-artifacts-local
aws --endpoint-url=http://localhost:4566 s3 mb s3://dvc-store

# Descargar datos y artefactos mediante DVC
dvc remote modify localstack endpointurl http://localhost:4566
dvc pull
```

### 3. Entrenamiento y Seguimiento del Modelo

Ejecuta el script de entrenamiento para realizar la optimización de hiperparámetros con Optuna y registrar métricas y artefactos en MLflow y LocalStack:

```bash
python src/train.py
```

Para inspeccionar los experimentos en la interfaz de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Accede al dashboard en:

```text
http://localhost:5000
```

### 4. Contenerización y Despliegue en Kubernetes

Apunta tu CLI local de Docker al daemon de Docker de Minikube, construye la imagen del contenedor de la API y aplica los manifiestos de Kubernetes:

```bash
# Iniciar el clúster de Minikube
minikube start

# Redirigir Docker al entorno de Minikube
eval $(minikube -p minikube docker-env)

# Construir la imagen Docker dentro de Minikube
docker build -t hotel-mlops-api:latest .

# Desplegar la aplicación y el servicio en Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verificar el estado de los pods
kubectl get pods

# Exponer la URL del servicio
minikube service api-service --url
```

### 5. Dashboard Interactivo con Streamlit

Inicia el dashboard web de Streamlit para interactuar con el endpoint del modelo desplegado mediante FastAPI:

```bash
streamlit run app/dashboard.py
```

Abre la siguiente URL en tu navegador:

```text
http://localhost:8501
```

Envía características de reservas y visualiza predicciones de segmentos de mercado en tiempo real.

---

## Integración Continua (CI/CD)

La Integración Continua se gestiona mediante **GitHub Actions** (`.github/workflows/ci-cd.yml`).

Al realizar un push de cambios o abrir un pull request hacia la rama `master`, el flujo de trabajo ejecuta automáticamente:

1. Configuración de un entorno Python 3.12.
2. Instalación de las dependencias necesarias.
3. Ejecución de pruebas unitarias mediante `pytest`.
4. Construcción de la imagen Docker para verificar la integridad del build.

---

## Pruebas

Ejecuta la suite de pruebas unitarias localmente para verificar los esquemas de los endpoints de la API y el manejo de respuestas:

```bash
pytest tests/
```

La suite de pruebas valida:

- Disponibilidad del endpoint (`/predict`).
- Esquemas de validación de payloads con Pydantic.
- Códigos de estado de respuesta y estructura de salida correctos.