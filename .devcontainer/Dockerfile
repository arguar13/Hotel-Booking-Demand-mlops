FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Instalar dependencias del sistema operativo.
# --no-install-recommends: sin esto apt arrastra decenas de paquetes sugeridos
# que nadie usa, engordando la imagen y su superficie de ataque.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client awscli \
    && rm -rf /var/lib/apt/lists/*

# La imagen base de devcontainers ya trae el usuario no-root `vscode`, que es
# con el que VS Code se conecta. Poetry se instala para ese usuario, no para
# root, para que el entorno virtual del proyecto le pertenezca.
USER vscode
ENV PATH="/home/vscode/.local/bin:$PATH"

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Configurar Poetry para que cree el entorno virtual dentro del proyecto
RUN poetry config virtualenvs.in-project true
