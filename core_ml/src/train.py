import os
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import structlog
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import features as features_module
from src.config_loader import load_config
from src.data_contracts import DataContractError, validate_processed
from src.data_processing import use_toy_data
from src.features import FeatureSpec, build_model_pipeline
from src.logging_setup import configure_logging
from src.monitoring.drift_check import PerformanceBaseline, build_reference_profile
from src.traceability import collect_traceability_tags

# The pipeline contains BookingFeatureBuilder, a class from this package. The
# API image does not ship core_ml's code, so pickling that class *by
# reference* (the default for importable modules) would make the model
# unloadable there with "No module named 'src'". Pickling this one module by
# value embeds the class in the artifact instead.
cloudpickle.register_pickle_by_value(features_module)

log = structlog.get_logger("hotel_mlops.train")

# What it takes to unpickle and run the logged pipeline, pinned to the versions
# that trained it. Declared rather than left to MLflow's inference, which
# reloads the model in a subprocess to guess (~30s per run) and picks up
# whatever else happens to be imported in the training process.
MODEL_REQUIREMENTS = ("scikit-learn", "pandas", "numpy", "cloudpickle")


def _model_requirements() -> list[str]:
    return [f"{package}=={package_version(package)}" for package in MODEL_REQUIREMENTS]


# scikit-learn's own RandomForest defaults (unlimited depth, one-sample
# leaves). Enqueued as the first Optuna trial - see train_pipeline().
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """The hyperparameter search space; it contains DEFAULT_PARAMS.

    Unlimited depth stays a candidate: on ~50k bookings with class weights,
    fully grown trees are a strong configuration, and a space that only
    offers shallow ones can make tuning *worse* than not tuning.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 20, 30, 40]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5]),
    }


def temporal_backtest(
    df: pd.DataFrame,
    spec: FeatureSpec,
    params: dict[str, Any],
    random_state: int,
    target_column: str,
) -> dict[str, Any] | None:
    """Fit on every arrival year but the last, score the last one.

    Returns None when the data covers a single year (nothing to hold out).
    """
    years = sorted(int(year) for year in df["arrival_date_year"].dropna().unique())
    if len(years) < 2:
        return None
    holdout_year = years[-1]
    past = df[df["arrival_date_year"] < holdout_year]
    future = df[df["arrival_date_year"] == holdout_year]

    pipeline = build_model_pipeline(spec, params, random_state)
    pipeline.fit(past[spec.input_columns], past[target_column])
    predicted = pipeline.predict(future[spec.input_columns])
    return {
        "train_years": [year for year in years if year < holdout_year],
        "holdout_year": holdout_year,
        "n_train": int(len(past)),
        "n_test": int(len(future)),
        "f1_weighted": float(f1_score(future[target_column], predicted, average="weighted")),
        "f1_macro": float(f1_score(future[target_column], predicted, average="macro")),
    }


class QualityGateError(Exception):
    """Raised when a trained model fails to meet the minimum quality bar.

    The run is still logged to MLflow for audit, but it is never aliased in
    the registry, so the serving API never picks it up.
    """


def train_pipeline() -> str:
    """
    Trains the ML pipeline, performs hyperparameter tuning with Optuna,
    and logs the best model to MLflow as the single, immutable source of
    truth for the model artifact (no loose model.joblib files).

    Returns:
        str: the MLflow run ID of the training run.
    """
    config = load_config()

    # MLflow setup. MLFLOW_TRACKING_URI overrides config.yaml, e.g. to point
    # a CI smoke run at a throwaway local SQLite store instead of a real server.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    # Which store this run (and the model version it registers) actually landed
    # in is the first thing you need when a run "disappears" - log it explicitly
    # rather than inferring it from config precedence.
    log.info(
        "mlflow_configured",
        tracking_uri=mlflow.get_tracking_uri(),
        registry_uri=mlflow.get_registry_uri(),
        experiment=config["mlflow"]["experiment_name"],
    )

    if use_toy_data():
        processed_path = config["data"]["toy_processed_data_path"]
        log.info("training_against_toy_dataset", processed_path=processed_path)
    else:
        processed_path = config["data"]["processed_data_path"]

    # Load data - fail fast if it doesn't satisfy the processed data contract,
    # before spending any compute on Optuna/model fitting.
    df = pd.read_csv(processed_path)
    df = validate_processed(df)

    # Optional training period. A model is only ever fitted on data up to some
    # point in time, and saying so explicitly is what lets the drift monitor's
    # baseline mean "the world as of <date>" rather than "whatever was in the
    # CSV". Also what makes a genuine drift backtest possible - see
    # config.yaml's train_max_year.
    train_max_year = config["data"].get("train_max_year")
    if train_max_year and "arrival_date_year" in df.columns:
        before = len(df)
        df = df[df["arrival_date_year"] <= int(train_max_year)]
        log.info(
            "training_period_applied",
            train_max_year=int(train_max_year),
            rows_kept=len(df),
            rows_dropped=before - len(df),
        )
        df = validate_processed(df)

    # Only the allowlisted input columns, never "everything but the target":
    # see config.yaml's `features:` block for what is excluded and why.
    spec = FeatureSpec.from_config(config)
    missing = [c for c in spec.input_columns if c not in df.columns]
    if missing:
        raise DataContractError(f"processed dataset is missing model input column(s): {missing}")

    X = df[spec.input_columns]
    y = df[config["model"]["target_column"]]
    random_state = config["model"]["random_state"]

    # Three-way split. Optuna's objective below is scored on X_val only - never
    # on X_test - because a hyperparameter search that gets to see the test set
    # is a search that overfits to it: fifty-odd trials each nudging toward
    # whatever happens to work on that particular sample stops being a
    # held-out estimate and starts being the metric that was searched for. The
    # final pipeline is refit on train+val once tuning is settled (no reason
    # to discard labelled data once it is no longer influencing which
    # hyperparameters get picked) and X_test is then touched exactly once, for
    # the number that actually goes into the quality gate and the monitoring
    # baseline.
    test_size = config["model"]["test_size"]
    validation_size = config["model"]["validation_size"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=config["model"]["random_state"],
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_size / (test_size + validation_size),
        random_state=config["model"]["random_state"],
        stratify=y_holdout,
    )

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_model_pipeline(spec, suggest_params(trial), random_state)
        pipeline.fit(X_train, y_train)
        return float(f1_score(y_val, pipeline.predict(X_val), average="weighted"))

    log.info("optuna_tuning_started", n_trials=config["model"]["n_trials_optuna"])
    # Seeded sampler: the same config explores the same trials, so two runs on
    # the same data and commit produce the same model.
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    # The untuned configuration is always trial 0, so tuning can only ever
    # replace it with something that scores better on validation - never
    # ship a "tuned" model that is worse than the defaults it started from.
    study.enqueue_trial(DEFAULT_PARAMS)
    study.optimize(objective, n_trials=config["model"]["n_trials_optuna"])

    best_params = study.best_params
    default_val_f1 = float(study.trials[0].value or 0.0)
    log.info(
        "optuna_tuning_finished",
        best_params=best_params,
        best_val_f1=study.best_value,
        default_val_f1=default_val_f1,
    )

    # Final Model Training with MLflow logging
    with mlflow.start_run(run_name="Best_RandomForest_Model") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("val_f1_best", float(study.best_value))
        mlflow.log_metric("val_f1_default", default_val_f1)
        mlflow.set_tags(collect_traceability_tags(processed_path))
        mlflow.set_tag("used_toy_data", str(use_toy_data()))
        mlflow.log_dict(
            {
                "input_columns": spec.input_columns,
                "numeric": spec.numeric,
                "cyclical": spec.cyclical,
                "presence_flags": spec.presence_flags,
                "categorical": spec.categorical,
            },
            "features/feature_spec.json",
        )

        final_pipeline = build_model_pipeline(spec, best_params, random_state)

        # Refit on train+val: validation's job (selecting best_params) is done,
        # so folding it back into the fit gives the shipped model more signal
        # without touching X_test, which still has never been seen by anything.
        X_fit = pd.concat([X_train, X_val])
        y_fit = pd.concat([y_train, y_val])
        final_pipeline.fit(X_fit, y_fit)
        y_pred = final_pipeline.predict(X_test)

        # Metrics. Weighted F1 drives the quality gate and promotion; macro F1
        # is logged next to it because the weighted average is dominated by
        # Online TA (~half the rows) and hides a model that gives up on the
        # small segments (Aviation, Complementary). The per-class report is
        # where to look when the two diverge.
        f1 = f1_score(y_test, y_pred, average="weighted")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_dict(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "evaluation/classification_report.json",
        )
        labels = [str(label) for label in final_pipeline.classes_]
        # Threshold-free view per segment: a class with high ROC AUC but low
        # recall is ranked well and only loses at the argmax decision - a
        # thresholding problem, not a feature problem.
        probabilities = final_pipeline.predict_proba(X_test)
        roc_auc = {
            label: float(roc_auc_score((y_test == label).astype(int), probabilities[:, i]))
            for i, label in enumerate(labels)
        }
        mlflow.log_dict(roc_auc, "evaluation/roc_auc_ovr.json")
        mlflow.log_metric("roc_auc_ovr_macro", float(sum(roc_auc.values()) / len(roc_auc)))
        mlflow.log_dict(
            {
                "labels": labels,
                "rows": "true class",
                "columns": "predicted class",
                "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            },
            "evaluation/confusion_matrix.json",
        )

        # A floor to read every other number against: always predicting the
        # most frequent segment already gets ~59% accuracy on this data, which
        # is why accuracy alone says little here. Scored on the same test split.
        naive = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)
        naive_pred = naive.predict(X_test)
        mlflow.log_metric("baseline_f1_score", f1_score(y_test, naive_pred, average="weighted"))
        mlflow.log_metric("baseline_f1_macro", f1_score(y_test, naive_pred, average="macro"))

        # The random split above estimates performance on bookings like the
        # ones trained on. Production only ever sees *later* bookings, so this
        # also refits on every year but the last and scores the last one - the
        # number to expect once the booking mix has moved on.
        if config["model"].get("temporal_backtest", True):
            backtest = temporal_backtest(
                df, spec, best_params, random_state, config["model"]["target_column"]
            )
            if backtest is not None:
                mlflow.log_metric("temporal_f1_score", backtest["f1_weighted"])
                mlflow.log_metric("temporal_f1_macro", backtest["f1_macro"])
                mlflow.log_dict(backtest, "evaluation/temporal_backtest.json")
                log.info("temporal_backtest_logged", **backtest)

        # --- Monitoring baseline -------------------------------------------
        # Drift is a comparison, so a model is only monitorable if the
        # distribution it was fitted on travels with it. This writes that
        # baseline - mean/std per numeric feature, plus the held-out
        # performance - as an artifact of this run, which makes it immutable
        # and reachable from the model version alone. The drift check
        # (src/monitoring/drift_check.py) reads it back through the registry
        # alias and never needs the training dataset, DVC, or bucket
        # credentials to do its job.
        #
        # Built from X_fit (train+val), not the full frame: the reference for
        # "what did this model learn from" is exactly the rows it was fitted
        # on, which is train+val now that val has been folded back in. Only
        # the raw numeric inputs the API receives - an identifier's mean
        # (agent, company) is not a distribution worth testing.
        reference_profile = build_reference_profile(
            features=X_fit[spec.monitored_numeric_columns],
            performance=PerformanceBaseline(
                f1_weighted=float(f1),
                accuracy=float(acc),
                n_eval=int(len(y_test)),
            ),
        )
        with tempfile.TemporaryDirectory() as staging_dir:
            profile_path = Path(staging_dir) / "reference_profile.json"
            reference_profile.write(profile_path)
            mlflow.log_artifact(str(profile_path), artifact_path="monitoring")
        log.info(
            "reference_profile_logged",
            numeric_features=len(reference_profile.numeric),
            baseline_f1=float(f1),
        )

        # Log the model as an immutable, versioned MLflow Registry entry -
        # this is the only place the trained artifact lives; no local
        # model.joblib is ever written.
        registry_name = config["model"]["registry_name"]
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            registered_model_name=registry_name,
            serialization_format="cloudpickle",
            pip_requirements=_model_requirements(),
        )

        run_id = run.info.run_id
        log.info("run_logged", run_id=run_id, f1_score=f1, f1_macro=f1_macro, accuracy=acc)

        # Quality gate: an absolute floor a version must clear to even be
        # *considered* for promotion. A run that fails it is still fully
        # logged (metrics, params, artifact, traceability tags) for audit -
        # it just can never reach promote_model.py's canary comparison,
        # let alone win it. This gate alone cannot tell a version that beats
        # the floor but is worse than what is already serving - that
        # comparison is promote_model.py's job, not this one's.
        min_f1 = config["model"]["min_f1_threshold"]
        registry_alias = config["model"]["registry_alias"]
        if f1 < min_f1:
            mlflow.set_tag("quality_gate", "failed")
            raise QualityGateError(
                f"Run {run_id} scored f1={f1:.4f}, below the minimum "
                f"threshold of {min_f1}. Not eligible for promotion to "
                f"'{registry_alias}'."
            )

        mlflow.set_tag("quality_gate", "passed")
        [registered_version] = MlflowClient().search_model_versions(f"run_id='{run_id}'")
        log.info(
            "model_version_registered",
            version=registered_version.version,
            registry_name=registry_name,
            next_step=(
                f"run `python -m src.promote_model --run-id {run_id}` to compare this "
                f"version against whatever is currently aliased '{registry_alias}' and, "
                "if it clears the canary bar, move the alias"
            ),
        )

        return run_id


if __name__ == "__main__":
    configure_logging()
    completed_run_id = train_pipeline()
    # structlog is configured above to print JSON lines to this same stdout
    # (PrintLoggerFactory), so the run id cannot just be `print()`-ed without
    # a caller having to pick it out of a log stream. A file is what lets
    # `make promote` (and .gitlab-ci.yml's manual `train` stage) chain
    # straight into `python -m src.promote_model --run-id $(cat run_id.txt)`
    # without parsing logs.
    Path("run_id.txt").write_text(completed_run_id, encoding="utf-8")
