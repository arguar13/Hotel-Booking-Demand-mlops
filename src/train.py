import pandas as pd
import logging
import optuna
import mlflow
import joblib
import os
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.config_loader import load_config

logger = logging.getLogger(__name__)

def train_pipeline():
    """
    Trains the ML pipeline, performs hyperparameter tuning with Optuna,
    and logs the best model using MLflow.
    """
    config = load_config()
    
    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Load data
    df = pd.read_csv(config["data"]["processed_data_path"])
    X = df.drop(columns=[config["model"]["target_column"]])
    y = df[config["model"]["target_column"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"], stratify=y
    )
    
    # Preprocessing definitions
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Numerical pipeline: Impute missing values with median, then scale
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing values with most frequent, then one-hot encode
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ])
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=config["model"]["random_state"]
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=config["model"]["random_state"])),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        return f1_score(y_test, preds, average='weighted')

    logger.info("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config["model"]["n_trials_optuna"])
    
    best_params = study.best_params
    logger.info(f"Best parameters found: {best_params}")
    
    # Final Model Training with MLflow logging
    with mlflow.start_run(run_name="Best_RandomForest_Model"):
        mlflow.log_params(best_params)
        
        final_model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=config["model"]["random_state"]
        )
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=config["model"]["random_state"])),
            ('classifier', final_model)
        ])
        
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        
        # Log Model to Registry
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name="HotelSegmentClassifier",
            serialization_format="cloudpickle"
        )
        
        # NUEVO: Guardar modelo localmente para DVC y Cloud Run
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(final_pipeline, model_path)
        
        logger.info(f"Model trained and logged to MLflow successfully. F1: {f1:.4f}")
        logger.info(f"Model saved locally for DVC tracking at {model_path}")

if __name__ == "__main__":
    train_pipeline()