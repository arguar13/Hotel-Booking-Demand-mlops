"""Traceability helpers.

Every training run must be reconstructible from a single tuple:
    Git commit hash + DVC data hash + hyperparameters + MLflow run ID + container image tag

Hyperparameters and the MLflow run ID are already first-class MLflow
concepts (`mlflow.log_params` / the active run's `run_id`). This module
supplies the other two: the Git commit the code was run at, and the DVC
content hash of the exact dataset that was read.
"""

import logging
import os
import subprocess  # nosec B404 - used below with a fixed argv and no shell
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"


def get_git_commit_hash() -> str:
    """Git commit hash of the current working tree.

    Prefers CI_COMMIT_SHA (set by GitLab CI) so the hash reflects the commit
    actually being built, then falls back to asking git directly for local runs.
    """
    ci_sha = os.getenv("CI_COMMIT_SHA")
    if ci_sha:
        return ci_sha

    try:
        # Fixed argv, no shell, no user-controlled input - safe despite bandit's
        # generic subprocess warnings.
        result = subprocess.run(  # nosec
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"Could not determine git commit hash: {e}")
        return UNKNOWN


def get_dvc_data_hash(data_path: str, dvc_lock_path: str = "dvc.lock") -> str:
    """MD5 content hash DVC assigned to `data_path`.

    This is the same hash `dvc status`/`dvc push` use, so a run tagged with
    it can always be traced back to the exact bytes of the dataset it saw,
    regardless of the current state of the working tree. Looks in two places,
    since DVC records hashes differently depending on how a path is tracked:

    1. `dvc.lock` - for paths produced by a `dvc.yaml` pipeline stage
       (e.g. the processed CSV emitted by the `clean_data`/`clean_data_toy` stages).
    2. `<data_path>.dvc` - for paths tracked directly with `dvc add`
       (e.g. the raw source CSVs).
    """
    lock_file = Path(dvc_lock_path)
    if lock_file.exists():
        try:
            lock = yaml.safe_load(lock_file.read_text()) or {}
            for stage in lock.get("stages", {}).values():
                for entry in [*stage.get("deps", []), *stage.get("outs", [])]:
                    if entry.get("path") == data_path:
                        return str(entry.get("md5", UNKNOWN))
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"Could not read {lock_file}: {e}")

    dvc_file = Path(f"{data_path}.dvc")
    if dvc_file.exists():
        try:
            meta = yaml.safe_load(dvc_file.read_text())
            outs = meta.get("outs", [])
            if outs:
                return str(outs[0].get("md5", UNKNOWN))
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"Could not read DVC hash from {dvc_file}: {e}")

    logger.warning(f"No DVC hash found for {data_path} in {lock_file} or {dvc_file}")
    return UNKNOWN


def get_container_image_tag() -> str:
    """Image tag this run executed under, or 'local-dev' outside CI/containers."""
    return os.getenv("CI_COMMIT_SHA") or os.getenv("IMAGE_TAG") or "local-dev"


def collect_traceability_tags(processed_data_path: str) -> dict[str, str]:
    """The three externally-supplied legs of the traceability tuple, as MLflow tags.

    The fourth and fifth legs (hyperparameters, MLflow run ID) are logged by
    `mlflow.log_params` and assigned by `mlflow.start_run()` respectively, so
    they don't need to be collected here.
    """
    return {
        "git_commit_hash": get_git_commit_hash(),
        "dvc_data_hash": get_dvc_data_hash(processed_data_path),
        "container_image_tag": get_container_image_tag(),
    }
