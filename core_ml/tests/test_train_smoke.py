"""Fast, hermetic end-to-end smoke test for the training pipeline.

Runs the full clean-contract-tune-fit-log-quality-gate loop against a tiny,
perfectly-separable synthetic dataset and a local SQLite-backed MLflow store
(no server, no S3, no real dataset needed) so it completes in a few seconds
and proves the pipeline's wiring - not the model's real-world accuracy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src import train as train_module
from src.train import QualityGateError, train_pipeline

N_PER_CLASS = 20
CLASS_LEAD_TIME_OFFSETS = {"Direct": 0, "Corporate": 100, "Groups": 200}


def _make_processed_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for segment, offset in CLASS_LEAD_TIME_OFFSETS.items():
        for _ in range(N_PER_CLASS):
            rows.append(
                {
                    "hotel": "City Hotel",
                    "market_segment": segment,
                    "month": 7,
                    "lead_time": offset + int(rng.integers(0, 10)),
                    "adults": 2,
                    "babies": 0,
                    "children": 0.0,
                    "adr": 100.0,
                    "booking_changes": 0,
                    "days_in_waiting_list": 0,
                    "total_of_special_requests": 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _isolate_mlflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep each test's MLflow store hermetic.

    `train_pipeline` deliberately lets MLFLOW_TRACKING_URI override config.yaml
    (that is how CI points a smoke run at a throwaway SQLite store). But MLflow
    itself *writes* MLFLOW_TRACKING_URI into os.environ while logging a model,
    so without this the first test leaks its store into every later one: the
    second test would register its model version into the first test's database
    and then assert against its own empty one.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)


@pytest.fixture
def base_config(tmp_path: Path) -> dict:
    processed_path = tmp_path / "processed.csv"
    _make_processed_df().to_csv(processed_path, index=False)

    return {
        "data": {
            "processed_data_path": str(processed_path),
        },
        "model": {
            "target_column": "market_segment",
            "test_size": 0.3,
            "validation_size": 0.2,
            "random_state": 42,
            "n_trials_optuna": 1,
            "min_f1_threshold": 0.5,
            "registry_name": "SmokeTestClassifier",
            "registry_alias": "staging",
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "experiment_name": "smoke_test",
        },
    }


def test_train_pipeline_registers_a_promotion_candidate_when_quality_gate_passes(
    base_config, monkeypatch
) -> None:
    """train.py registers and tags the version; it no longer moves the alias.

    Promotion is promote_model.py's job (see that module's docstring for
    why: an absolute quality gate alone cannot tell a version that clears
    the floor but is worse than what is already serving). This test
    documents the boundary - the companion end-to-end test in
    test_promote_model.py covers train_pipeline() followed by promote().
    """
    monkeypatch.setattr(train_module, "load_config", lambda: base_config)
    monkeypatch.setattr(train_module, "use_toy_data", lambda: False)

    run_id = train_pipeline()

    assert run_id

    client = MlflowClient(tracking_uri=base_config["mlflow"]["tracking_uri"])
    registry_name = base_config["model"]["registry_name"]
    registry_alias = base_config["model"]["registry_alias"]

    versions = client.search_model_versions(f"name='{registry_name}'")
    assert len(versions) == 1
    assert versions[0].run_id == run_id

    run = client.get_run(run_id)
    assert run.data.tags.get("quality_gate") == "passed"

    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(registry_name, registry_alias)


def test_train_pipeline_fails_quality_gate_without_promoting(base_config, monkeypatch) -> None:
    base_config["model"]["min_f1_threshold"] = 1.1  # > 1.0: unreachable by any F1 score
    monkeypatch.setattr(train_module, "load_config", lambda: base_config)
    monkeypatch.setattr(train_module, "use_toy_data", lambda: False)

    with pytest.raises(QualityGateError):
        train_pipeline()

    client = MlflowClient(tracking_uri=base_config["mlflow"]["tracking_uri"])
    registry_name = base_config["model"]["registry_name"]
    registry_alias = base_config["model"]["registry_alias"]

    # The run is still logged (registered) for audit, but never aliased.
    versions = client.search_model_versions(f"name='{registry_name}'")
    assert len(versions) == 1
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(registry_name, registry_alias)
