FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

WORKDIR /app
ENV PYTHONPATH="/app/api:${PYTHONPATH}"

# Optional local CA trust for machines where a corporate/AV proxy performs
# TLS interception on outbound HTTPS (see README "Notas y solución de
# problemas" / GUIA_EJECUCION_PROYECTO). `docker/local-c[a].crt` is a glob
# (not a literal path) so COPY silently matches nothing when the file is
# absent - this needs to work identically whether the image is built via
# `docker build`, `docker compose build`, or testcontainers' DockerImage
# (plain Docker Engine build API, no BuildKit secret support), so a plain
# COPY is used instead of a `--mount=type=secret`. A CA certificate is
# public data (no private key), so baking it into a layer is not a secrets
# leak. No-op everywhere else, including CI, where only the committed empty
# placeholder exists.
COPY docker/local-ca-placeholder.crt docker/local-c[a].crt /tmp/local-ca/
RUN sh -c 'for f in /tmp/local-ca/*; do [ -s "$f" ] && cp "$f" /usr/local/share/ca-certificates/local-ca.crt; done; update-ca-certificates; rm -rf /tmp/local-ca'
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN pip install poetry==$POETRY_VERSION

COPY api/pyproject.toml api/poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

COPY api/ ./api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
