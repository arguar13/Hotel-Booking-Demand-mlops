"""Fast, hermetic end-to-end smoke test for the training pipeline.

Runs the full clean-contract-tune-fit-log-quality-gate loop against a tiny,
perfectly-separable synthetic dataset and a local SQLite-backed MLflow store
(no server, no S3, no real dataset needed) so it completes in a few seconds
and proves the pipeline's wiring - not the model's real-world accuracy.
"""

from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src import train as train_module
from src.config_loader import load_config
from src.features import FeatureSpec
from src.train import QualityGateError, temporal_backtest, train_pipeline

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
                    "arrival_date_year": 2016,
                    "month": 7,
                    "arrival_date_week_number": 27,
                    "arrival_date_day_of_month": 1,
                    "lead_time": offset + int(rng.integers(0, 10)),
                    "stays_in_weekend_nights": 1,
                    "stays_in_week_nights": 2,
                    "adults": 2,
                    "babies": 0,
                    "children": 0.0,
                    "meal": "BB",
                    "country": "PRT",
                    "is_repeated_guest": 0,
                    "previous_cancellations": 0,
                    "previous_bookings_not_canceled": 0,
                    "reserved_room_type": "A",
                    "deposit_type": "No Deposit",
                    "customer_type": "Transient",
                    "adr": 100.0,
                    "days_in_waiting_list": 0,
                    "required_car_parking_spaces": 0,
                    "total_of_special_requests": 0,
                    "agent": 9.0 if segment != "Direct" else float("nan"),
                    "company": float("nan"),
                    # Post-booking columns that are present in the processed
                    # CSV but must never be used as model inputs.
                    "distribution_channel": segment,
                    "reservation_status": "Check-Out",
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
        "features": load_config()["features"],
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
    assert "f1_macro" in run.data.metrics
    # The untuned default is always a candidate, so tuning never loses to it.
    assert run.data.metrics["val_f1_best"] >= run.data.metrics["val_f1_default"]
    assert "baseline_f1_macro" in run.data.metrics
    assert 0.0 <= run.data.metrics["roc_auc_ovr_macro"] <= 1.0
    confusion = mlflow.artifacts.load_dict(f"runs:/{run_id}/evaluation/confusion_matrix.json")
    assert sum(map(sum, confusion["matrix"])) > 0
    # The synthetic frame covers a single arrival year: nothing to hold out.
    assert "temporal_f1_score" not in run.data.metrics

    # The processed frame carries distribution_channel / reservation_status;
    # the model must have been trained on the allowlist only.
    feature_spec = mlflow.artifacts.load_dict(f"runs:/{run_id}/features/feature_spec.json")
    assert "distribution_channel" not in feature_spec["input_columns"]
    assert "reservation_status" not in feature_spec["input_columns"]

    # Pinned, declared requirements - not whatever MLflow inferred.
    requirements_file = mlflow.pyfunc.get_model_dependencies(f"models:/{registry_name}/1")
    requirements = Path(requirements_file).read_text(encoding="utf-8")
    assert "scikit-learn==" in requirements
    assert "imbalanced-learn" not in requirements

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


def test_temporal_backtest_holds_out_the_last_year() -> None:
    frame = pd.concat(
        [_make_processed_df().assign(arrival_date_year=year) for year in (2015, 2016, 2017)],
        ignore_index=True,
    )
    spec = FeatureSpec.from_config(load_config())

    result = temporal_backtest(frame, spec, {"n_estimators": 10}, 0, "market_segment")

    assert result is not None
    assert result["train_years"] == [2015, 2016]
    assert result["holdout_year"] == 2017
    assert result["n_test"] == len(frame) // 3
    assert 0.0 <= result["f1_macro"] <= 1.0


def test_temporal_backtest_skips_a_single_year() -> None:
    spec = FeatureSpec.from_config(load_config())

    assert temporal_backtest(_make_processed_df(), spec, {}, 0, "market_segment") is None
