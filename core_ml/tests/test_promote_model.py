"""promote_model.py against a real, hermetic MLflow store (local SQLite, no server).

Each test logs candidate "training runs" directly - a dummy sklearn model
plus a monitoring/reference_profile.json artifact with a chosen
performance.f1_weighted - rather than running the full train_pipeline(),
so the canary comparison's f1 values are exact and deterministic. The
companion smoke test in test_train_smoke.py already covers that train.py
itself produces a run in this same shape.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from src.monitoring.profile import PerformanceBaseline, ReferenceProfile
from src.promote_model import PromotionError, promote, rollback

REGISTRY_NAME = "PromoteTestClassifier"
REGISTRY_ALIAS = "staging"


@pytest.fixture(autouse=True)
def _isolate_mlflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)


@pytest.fixture
def config(tmp_path: Path) -> dict:
    return {
        "model": {
            "registry_name": REGISTRY_NAME,
            "registry_alias": REGISTRY_ALIAS,
            "canary_tolerance": -0.01,
        },
        "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"},
    }


def _log_candidate_run(config: dict, f1_weighted: float, quality_gate: str = "passed") -> str:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("promote_model_tests")
    with mlflow.start_run() as run:
        model = DummyClassifier(strategy="constant", constant=0).fit([[0.0]], [0])
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=config["model"]["registry_name"],
        )
        mlflow.set_tag("quality_gate", quality_gate)

        profile = ReferenceProfile(
            created_at="2026-01-01T00:00:00+00:00",
            n_rows=100,
            target_column="market_segment",
            n_bins=10,
            performance=PerformanceBaseline(
                f1_weighted=f1_weighted,
                accuracy=f1_weighted,
                mean_confidence=0.9,
                mean_margin=0.5,
                n_eval=100,
            ),
        )
        with tempfile.TemporaryDirectory() as staging_dir:
            path = Path(staging_dir) / "reference_profile.json"
            profile.write(path)
            mlflow.log_artifact(str(path), artifact_path="monitoring")

        return run.info.run_id


def test_first_candidate_promotes_with_no_champion_to_compare_against(config: dict) -> None:
    run_id = _log_candidate_run(config, f1_weighted=0.70)

    version = promote(run_id, config)

    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    aliased = client.get_model_version_by_alias(REGISTRY_NAME, REGISTRY_ALIAS)
    assert str(aliased.version) == version
    assert aliased.tags["promoted_from_version"] == "none"


def test_a_challenger_that_beats_the_champion_is_promoted(config: dict) -> None:
    champion_run = _log_candidate_run(config, f1_weighted=0.70)
    champion_version = promote(champion_run, config)

    challenger_run = _log_candidate_run(config, f1_weighted=0.80)
    challenger_version = promote(challenger_run, config)

    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    aliased = client.get_model_version_by_alias(REGISTRY_NAME, REGISTRY_ALIAS)
    assert str(aliased.version) == challenger_version
    assert str(aliased.version) != champion_version
    assert aliased.tags["promoted_from_version"] == champion_version


def test_a_challenger_within_tolerance_still_promotes(config: dict) -> None:
    """canary_tolerance=-0.01: a small regression is still allowed to ship."""
    champion_run = _log_candidate_run(config, f1_weighted=0.70)
    promote(champion_run, config)

    challenger_run = _log_candidate_run(config, f1_weighted=0.695)  # 0.005 below, within tolerance
    challenger_version = promote(challenger_run, config)

    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    aliased = client.get_model_version_by_alias(REGISTRY_NAME, REGISTRY_ALIAS)
    assert str(aliased.version) == challenger_version


def test_a_challenger_that_regresses_beyond_tolerance_is_rejected(config: dict) -> None:
    champion_run = _log_candidate_run(config, f1_weighted=0.70)
    champion_version = promote(champion_run, config)

    challenger_run = _log_candidate_run(config, f1_weighted=0.50)  # far worse

    with pytest.raises(PromotionError, match="refusing to promote"):
        promote(challenger_run, config)

    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    aliased = client.get_model_version_by_alias(REGISTRY_NAME, REGISTRY_ALIAS)
    # The champion must still be the one serving - a rejection changes nothing.
    assert str(aliased.version) == champion_version

    run = client.get_run(challenger_run)
    assert run.data.tags.get("canary_verdict") == "rejected"


def test_a_run_that_never_passed_the_quality_gate_cannot_be_promoted(config: dict) -> None:
    run_id = _log_candidate_run(config, f1_weighted=0.90, quality_gate="failed")

    with pytest.raises(PromotionError, match="quality_gate=passed"):
        promote(run_id, config)


def test_rollback_moves_the_alias_back_exactly_one_promotion(config: dict) -> None:
    first_run = _log_candidate_run(config, f1_weighted=0.70)
    first_version = promote(first_run, config)

    second_run = _log_candidate_run(config, f1_weighted=0.80)
    second_version = promote(second_run, config)

    restored_version = rollback(config)

    assert restored_version == first_version
    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    aliased = client.get_model_version_by_alias(REGISTRY_NAME, REGISTRY_ALIAS)
    assert str(aliased.version) == first_version
    assert str(aliased.version) != second_version


def test_rollback_with_no_prior_promotion_fails_loudly(config: dict) -> None:
    run_id = _log_candidate_run(config, f1_weighted=0.70)
    promote(run_id, config)  # bootstrap promotion - promoted_from_version is "none"

    with pytest.raises(PromotionError, match="Nothing to roll back to"):
        rollback(config)


def test_rollback_without_any_aliased_version_raises(config: dict) -> None:
    with pytest.raises(MlflowException):
        rollback(config)
