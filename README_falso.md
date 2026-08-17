*Read this in other languages: [Español](README_es.md)*

# End-to-End MLOps Pipeline: Hotel Booking Market Segment Prediction

An end-to-end Machine Learning Operations (MLOps) pipeline designed to predict market segments for hotel bookings. This project covers the complete lifecycle of a machine learning application: from data versioning and hyperparameter optimization to containerization, Kubernetes orchestration, continuous integration, and real-time model serving.

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Local Setup and Execution](#local-setup-and-execution)
  - [1. Clone Repository and Environment Setup](#1-clone-repository-and-environment-setup)
  - [2. LocalStack Configuration and DVC Data Pull](#2-localstack-configuration-and-dvc-data-pull)
  - [3. Model Training and Tracking](#3-model-training-and-tracking)
  - [4. Containerization and Kubernetes Deployment](#4-containerization-and-kubernetes-deployment)
  - [5. Streamlit Interactive Dashboard](#5-streamlit-interactive-dashboard)
- [Continuous Integration (CI/CD)](#continuous-integration-cicd)
- [Testing](#testing)

---

## Overview

This repository demonstrates a production-ready MLOps framework that automates model training, tracking, version control, and containerized deployment. The goal is to provide a reproducible, tested, and scalable machine learning service using local cloud simulations to ensure zero cloud infrastructure cost during development.

### Key Capabilities

- Reproducible data and model versioning utilizing DVC backed by LocalStack S3.
- Automated hyperparameter search with Optuna and experiment tracking with MLflow.
- REST API implementation using FastAPI with full request validation via Pydantic.
- Containerized deployment using Docker and Kubernetes (Minikube).
- Continuous Integration via GitHub Actions running automated unit testing and Docker build checks.
- User interface for model inference powered by Streamlit.

---

## Business Problem

In the hospitality industry, understanding booking sources and market segments (e.g., Direct, Corporate, Online Travel Agents) is critical for optimizing revenue management, resource allocation, and marketing strategies.

This pipeline processes historical hotel booking features—such as lead time, length of stay, customer attributes, and room types—to classify and predict the market segment of incoming reservations in real time.

---

## System Architecture

The following diagram illustrates the flow of data, tracking, containerization, and deployment across the pipeline:

```text
                                 +-----------------------------------+
                                 |          Git Repository           |
                                 +-----------------+-----------------+
                                                   |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
       +-------------------+                             +-------------------+
       |    DVC Pointers   |                             |  GitHub Actions   |
       |                   |                             |      (CI/CD)      |
       +---------+---------+                             +---------+---------+
                 |                                                 |
                 v                                                 v
       +-------------------+                             +-------------------+
       |   LocalStack S3   |                             |  Pytest & Docker  |
       |  (s3://dvc-store) |                             | Build Verification|
       +-------------------+                             +-------------------+

                         +-----------------------------------+
                         |         Training Pipeline         |
                         | (Optuna Tuning + MLflow Tracking) |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |          LocalStack S3            |
                         |        (MLflow Artifacts)         |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |            FastAPI App            |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |         Docker Container          |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |        Kubernetes Cluster         |
                         |            (Minikube)             |
                         +-----------------+-----------------+
                                           ^
                                           |
                         +-----------------+-----------------+
                         |           Streamlit UI            |
                         +-----------------------------------+
```

---

## Tech Stack

| Category | Technology |
|-----------|------------|
| **Language & Core** | Python 3.12, Pandas, Scikit-learn, Imbalanced-learn |
| **Optimization & Tracking** | Optuna, MLflow |
| **Storage & Versioning** | DVC (Data Version Control), LocalStack (AWS S3 Emulation), Git |
| **API & Validation** | FastAPI, Pydantic, Uvicorn |
| **User Interface** | Streamlit |
| **Containerization & Orchestration** | Docker, Kubernetes (Minikube), Kubectl |
| **CI/CD & Testing** | GitHub Actions, Pytest, HTTPX |

---

## Repository Structure

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

## Prerequisites

Ensure the following tools are installed on your local system:

- **Python 3.12+**
- **Git**
- **Docker Desktop**
- **Minikube**
- **Kubectl**
- **AWS CLI** (for interacting with LocalStack)

---

## Local Setup and Execution

### 1. Clone Repository and Environment Setup

Clone the repository and set up a Python virtual environment:

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

### 2. LocalStack Configuration and DVC Data Pull

Start LocalStack and create the required S3 buckets for artifact storage and data versioning:

```bash
# Ensure LocalStack container is running on port 4566
docker run -d -p 4566:4566 -p 4571:4571 localstack/localstack

# Create S3 buckets
aws --endpoint-url=http://localhost:4566 s3 mb s3://mlflow-artifacts-local
aws --endpoint-url=http://localhost:4566 s3 mb s3://dvc-store

# Pull data and model artifacts via DVC
dvc remote modify localstack endpointurl http://localhost:4566
dvc pull
```

### 3. Model Training and Tracking

Run the model training script to execute hyperparameter tuning via Optuna and log metrics/artifacts to MLflow and LocalStack:

```bash
python src/train.py
```

To inspect experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access the dashboard at:

```text
http://localhost:5000
```

### 4. Containerization and Kubernetes Deployment

Point your local Docker CLI to Minikube's Docker daemon, build the API container image, and apply the Kubernetes manifests:

```bash
# Start Minikube cluster
minikube start

# Direct Docker environment to Minikube
eval $(minikube -p minikube docker-env)

# Build Docker image inside Minikube environment
docker build -t hotel-mlops-api:latest .

# Deploy application and service to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify pod status
kubectl get pods

# Expose service URL
minikube service api-service --url
```

### 5. Streamlit Interactive Dashboard

Launch the Streamlit web dashboard to interact with the deployed FastAPI model endpoint:

```bash
streamlit run app/dashboard.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

Submit booking features and view market segment predictions in real time.

---

## Continuous Integration (CI/CD)

Continuous Integration is managed via **GitHub Actions** (`.github/workflows/ci-cd.yml`).

Upon pushing changes or opening a pull request to the `master` branch, the workflow automatically:

1. Sets up a Python 3.12 environment.
2. Installs required dependencies.
3. Executes unit tests via `pytest`.
4. Builds the Docker container image to verify build integrity.

---

## Testing

Execute the unit test suite locally to verify API endpoint schemas and response handling:

```bash
pytest tests/
```

The test suite validates:

- Endpoint availability (`/predict`).
- Pydantic payload validation schemas.
- Correct response status codes and output structure.