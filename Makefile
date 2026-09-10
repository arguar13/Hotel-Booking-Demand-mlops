# Single entry point for local development and CI.
# The same `make` targets run here and in .gitlab-ci.yml, against the exact
# same poetry.lock-pinned tool versions.
#
# Requires: python3.12, poetry, make. On Windows this expects Git Bash
# (already the case for this repo) as make's shell.
SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECTS := api core_ml

.PHONY: help install format lint typecheck test test-api test-core-ml \
        precommit precommit-install clean up down data-toy dvc-repro \
        dvc-pull dvc-push train train-toy drift-check replay replay-drift \
        promote docker-build docker-push deploy tf-fmt tf-validate tf-plan tf-apply

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install both poetry environments and register git hooks
	@for p in $(PROJECTS); do \
		echo "==> poetry install ($$p)"; \
		poetry -C $$p install --no-interaction || exit 1; \
	done
	$(MAKE) precommit-install

format: ## Auto-format code with ruff in every sub-project
	@for p in $(PROJECTS); do \
		echo "==> ruff format ($$p)"; poetry -C $$p run ruff format . || exit 1; \
		echo "==> ruff check --fix ($$p)"; poetry -C $$p run ruff check --fix . || exit 1; \
	done

lint: ## Run ruff (lint) and mypy in every sub-project
	@for p in $(PROJECTS); do \
		echo "==> ruff ($$p)"; poetry -C $$p run ruff check . || exit 1; \
	done
	$(MAKE) typecheck

typecheck: ## Run mypy in every sub-project
	@for p in $(PROJECTS); do \
		echo "==> mypy ($$p)"; poetry -C $$p run mypy . || exit 1; \
	done

test-api: ## Run the api/ test suite
	poetry -C api run pytest --cov=. --cov-report=term-missing

test-core-ml: ## Run the core_ml/ test suite
	poetry -C core_ml run pytest --cov=src --cov-report=term-missing

test: test-api test-core-ml ## Run both test suites

precommit: ## Run every pre-commit hook against all files
	pre-commit run --all-files

precommit-install: ## Register the git hooks defined in .pre-commit-config.yaml
	pre-commit install --install-hooks

clean: ## Remove caches and build artifacts
	@find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' \) -not -path '*/.git/*' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf api/dist core_ml/dist .coverage htmlcov

up: ## Start the full local stack (db, localstack, mlflow, api, dashboard)
	docker compose up --build

down: ## Stop the local docker-compose stack
	docker compose down

# --- Data / training ---

data-toy: ## Regenerate the ~1000-row DVC-tracked toy dataset from the raw CSV
	poetry -C core_ml run python scripts/make_toy_dataset.py

dvc-repro: ## Re-run the DVC data-cleaning pipeline (full + toy), refreshing dvc.lock
	poetry -C core_ml run dvc repro

dvc-pull: ## Pull DVC-tracked datasets from the S3 remote
	poetry -C core_ml run dvc pull

dvc-push: ## Push DVC-tracked datasets to the S3 remote
	poetry -C core_ml run dvc push

train: dvc-repro ## Train on the full dataset (needs a reachable MLflow server, e.g. `make up`)
	poetry -C core_ml run python -m src.train

train-toy: ## Fast (~seconds) end-to-end training run against the toy dataset
	poetry -C core_ml run dvc repro clean_data_toy
	HOTEL_MLOPS_USE_TOY_DATA=true poetry -C core_ml run python -m src.train

promote: ## Compare core_ml/run_id.txt's run against the current champion and promote if it clears the bar (use ARGS="--rollback" to revert instead)
	@if echo "$(ARGS)" | grep -q -- "--rollback"; then \
		poetry -C core_ml run python -m src.promote_model --rollback; \
	else \
		poetry -C core_ml run python -m src.promote_model --run-id "$$(cat core_ml/run_id.txt)"; \
	fi

# --- Monitoring / drift ---
# The loop, runnable end to end on a laptop: `make up`, train a model, replay
# a slice of real bookings through the API (which logs every prediction to
# Postgres), then run the same drift check the production CronJob runs, from
# the same image.

replay: ## Replay real 2016 bookings through /predict + /feedback (baseline period)
	poetry -C core_ml run python scripts/replay_bookings.py --year 2016 --limit 3000

replay-drift: ## Replay real 2017 bookings - the period where the channel mix actually shifted
	poetry -C core_ml run python scripts/replay_bookings.py --year 2017 --months 5 6 7 --limit 3000

drift-check: ## Run the drift check against the local stack, from the real jobs image
	docker compose --profile jobs run --rm --build drift-check

# --- Docker images ---

docker-build: ## Build the api, dashboard and jobs images locally
	docker build -t hotel-mlops-api:local -f Dockerfile .
	docker build -t hotel-mlops-dashboard:local -f Dockerfile.dashboard .
	docker build -t hotel-mlops-jobs:local -f Dockerfile.jobs .

docker-push: ## Push the three images built above (expects them already tagged for your registry)
	docker push hotel-mlops-api:local
	docker push hotel-mlops-dashboard:local
	docker push hotel-mlops-jobs:local

# --- Kubernetes ---

deploy: ## Render and apply the production Kustomize overlay against whatever cluster kubectl is currently pointed at
	kubectl apply -k kubernetes/overlays/production

k8s-build: ## Render the production Kustomize overlay (pure client-side, no cluster needed)
	kubectl kustomize kubernetes/overlays/production

# --- Terraform ---

tf-fmt: ## Check Terraform formatting
	terraform -chdir=terraform fmt -check -diff -recursive

tf-validate: ## Validate Terraform syntax and internal consistency
	terraform -chdir=terraform validate

tf-plan: ## Show what Terraform would change (needs AWS credentials + initialized backend)
	terraform -chdir=terraform plan

tf-apply: ## Apply the Terraform plan (needs AWS credentials + initialized backend)
	terraform -chdir=terraform apply
