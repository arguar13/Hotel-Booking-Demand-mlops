# Single entry point for local development and CI.
# Every quality gate (format, lint, typecheck, test, security) is invoked
# through here so a developer's laptop and the GitLab CI runner execute the
# exact same commands against the exact same poetry.lock-pinned tool versions.
#
# Requires: python3.12, poetry, make. On Windows this expects Git Bash
# (already the case for this repo) as make's shell.
SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECTS := api core_ml
ALL_PROJECTS := api core_ml integration-tests

.PHONY: help install lock format lint typecheck test test-api test-core-ml \
        test-integration security yaml-lint precommit precommit-install ci \
        clean up down data-toy dvc-repro dvc-pull dvc-push train train-toy \
        ci-dry-run ci-dry-run-train \
        runner-register runner-start runner-stop k8s-build tf-fmt \
        tf-validate tf-plan

GITLAB_CI_LOCAL_VERSION := 4.75.1

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install all poetry environments and register git hooks
	@for p in $(ALL_PROJECTS); do \
		echo "==> poetry install ($$p)"; \
		poetry -C $$p install --no-interaction || exit 1; \
	done
	$(MAKE) precommit-install

lock: ## Regenerate all poetry.lock files deterministically
	@for p in $(ALL_PROJECTS); do \
		echo "==> poetry lock ($$p)"; \
		poetry -C $$p lock || exit 1; \
	done

format: ## Auto-format code (black + isort) in every sub-project
	@for p in $(ALL_PROJECTS); do \
		echo "==> isort ($$p)"; poetry -C $$p run isort . || exit 1; \
		echo "==> black ($$p)"; poetry -C $$p run black . || exit 1; \
	done

lint: ## Run ruff (static lint) in every sub-project
	@for p in $(ALL_PROJECTS); do \
		echo "==> ruff ($$p)"; poetry -C $$p run ruff check . || exit 1; \
	done

typecheck: ## Run mypy in every sub-project
	@for p in $(ALL_PROJECTS); do \
		echo "==> mypy ($$p)"; poetry -C $$p run mypy . || exit 1; \
	done

test-api: ## Run the api/ test suite
	poetry -C api run pytest --cov=. --cov-report=term-missing

test-core-ml: ## Run the core_ml/ test suite
	poetry -C core_ml run pytest --cov=src --cov-report=term-missing

test: test-api test-core-ml ## Run the fast unit-test suites (no Docker required)

test-integration: ## Run the Testcontainers integration suite (needs a running Docker daemon; ~3-5 min)
	poetry -C integration-tests run pytest -v

security: ## Run bandit (SAST) in every sub-project + trivy IaC/secret scan
	@for p in $(PROJECTS); do \
		echo "==> bandit ($$p)"; poetry -C $$p run bandit -q -r . -x "./tests" || exit 1; \
	done
	@echo "==> bandit (integration-tests)"; poetry -C integration-tests run bandit -q -r . --skip B101,B105,B106 || exit 1
	@if command -v trivy >/dev/null 2>&1; then \
		echo "==> trivy fs (secrets + vulnerabilities)"; \
		trivy fs --exit-code 1 --severity HIGH,CRITICAL --skip-dirs '**/node_modules' --skip-dirs core_ml/data --skip-dirs core_ml/mlruns .; \
		echo "==> trivy config (Dockerfiles, Kubernetes manifests, Terraform)"; \
		trivy config --exit-code 1 --severity HIGH,CRITICAL --skip-dirs terraform/.terraform .; \
	else \
		echo "!! trivy not installed locally - skipping IaC/vuln scan (install: https://aquasecurity.github.io/trivy, or rely on the CI security job)"; \
	fi

yaml-lint: ## Lint all YAML manifests (k8s, CI, docker-compose)
	# Invoke core_ml's yamllint binary directly rather than via
	# `poetry -C core_ml run`: on Poetry >=2.0, `-C <dir> run <cmd>` executes
	# <cmd> with <dir> as its cwd (not the directory `make` was invoked
	# from), which would resolve ".yamllint.yml ." against core_ml/ instead
	# of the repo root and silently ignore/miss files.
	@venv="$$(poetry -C core_ml env info --path)"; \
	if [ -x "$$venv/bin/yamllint" ]; then bin="$$venv/bin/yamllint"; else bin="$$venv/Scripts/yamllint"; fi; \
	"$$bin" -c .yamllint.yml .

precommit: ## Run every pre-commit hook against all files
	pre-commit run --all-files

precommit-install: ## Register the git hooks defined in .pre-commit-config.yaml
	pre-commit install --install-hooks

ci: lint typecheck test security yaml-lint ## Full quality gate, exactly as run in GitLab CI

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

clean: ## Remove caches and build artifacts
	@find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' \) -not -path '*/.git/*' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf api/dist core_ml/dist .coverage htmlcov

up: ## Start the full local stack (db, localstack, kafka, mlflow, api, dashboard)
	docker compose up --build

down: ## Stop the local docker-compose stack
	docker compose down

# --- Local CI validation ---
# Two layers, cheapest first:
#   1. ci-dry-run*    - gitlab-ci-local, pure Docker on this machine, never
#                        talks to GitLab. Zero minutes, zero runner needed.
#   2. runner-register/-start - a real GitLab Runner (docker-compose
#                        `ci-local` profile) registered against this
#                        project. Every job in .gitlab-ci.yml is pinned to
#                        it via `tags: ["local-hardware"]`, so pushes to main are
#                        executed on this hardware, never GitLab.com's
#                        shared runners - but that also means a push
#                        stays "pending" forever if this runner isn't
#                        registered and running (`make runner-start`).

# MSYS_NO_PATHCONV=1: on Windows/Git Bash, MSYS auto-rewrites POSIX-looking
# args (e.g. /builds/...) into Windows paths before they reach Docker,
# which breaks the container workdir/volumes gitlab-ci-local sets up.
# No-op on Linux/macOS. --privileged: integration-tests' docker:24.0.5-dind
# service needs it to start its own inner dockerd - without it the service
# container fails during its iptables/mount setup and the job's
# `docker` hostname never resolves. CI_COMMIT_BRANCH=main overrides the
# rules: `if: $CI_COMMIT_BRANCH == "main"` guard on every job, which
# gitlab-ci-local otherwise derives from whatever local branch is checked
# out - this makes it run regardless of which branch you're actually on.
ci-dry-run: ## Simulate quality-gate/trivy-scan/integration-tests locally with gitlab-ci-local (0 GitLab minutes, needs Docker)
	MSYS_NO_PATHCONV=1 npx --yes gitlab-ci-local@$(GITLAB_CI_LOCAL_VERSION) --privileged --variable CI_COMMIT_BRANCH=main quality-gate trivy-scan integration-tests

ci-dry-run-train: ## Simulate train-smoke locally (needs AWS creds in .gitlab-ci-local-variables.yml, see .example)
	MSYS_NO_PATHCONV=1 npx --yes gitlab-ci-local@$(GITLAB_CI_LOCAL_VERSION) --variable CI_COMMIT_BRANCH=main train-smoke

# GitLab's newer runner-authentication-token flow (Settings > CI/CD > Runners
# > New project runner > create it there, with tag "local-hardware" set in that form)
# moved tags/description/locked/etc. server-side: passing --tag-list (or
# --description) here is now a hard registration error, not just ignored
# ("Runner configuration ... is reserved ... specified on the GitLab
# server"). Make sure the runner you create in the UI carries the "local"
# tag - .gitlab-ci.yml's tags: ["local-hardware"] on every job depends on it.
runner-register: ## Register a real local GitLab Runner against this project (needs GITLAB_URL + GITLAB_RUNNER_TOKEN env vars, token from Settings > CI/CD > Runners > New project runner)
	@test -n "$$GITLAB_URL" || (echo "Set GITLAB_URL (e.g. https://gitlab.com)" && exit 1)
	@test -n "$$GITLAB_RUNNER_TOKEN" || (echo "Set GITLAB_RUNNER_TOKEN (Settings > CI/CD > Runners > New project runner - authentication token)" && exit 1)
	docker compose --profile ci-local run --rm gitlab-runner register \
		--non-interactive \
		--url "$$GITLAB_URL" \
		--token "$$GITLAB_RUNNER_TOKEN" \
		--executor docker \
		--docker-image docker:24.0.5 \
		--docker-privileged=true

runner-start: ## Start the registered local runner, so it picks up real pipeline jobs on this machine
	docker compose --profile ci-local up gitlab-runner

runner-stop: ## Stop the local runner
	docker compose --profile ci-local down gitlab-runner

# --- Kubernetes / GitOps ---

k8s-build: ## Render the production Kustomize overlay (pure client-side, no cluster needed)
	kubectl kustomize kubernetes/overlays/production

# --- Terraform ---

tf-fmt: ## Check Terraform formatting
	terraform -chdir=terraform fmt -check -diff -recursive

tf-validate: ## Validate Terraform syntax and internal consistency
	terraform -chdir=terraform validate

tf-plan: ## Show what Terraform would change (needs AWS credentials + initialized backend)
	terraform -chdir=terraform plan
