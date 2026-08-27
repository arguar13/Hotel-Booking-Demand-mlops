FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Instalar dependencias del sistema operativo
RUN apt-get update && apt-get install -y postgresql-client awscli && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configurar Poetry para que cree el entorno virtual dentro del proyecto
RUN poetry config virtualenvs.in-project true
