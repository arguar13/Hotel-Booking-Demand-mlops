import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from api.schemas import BookingFeatures
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hotel Market Segmentation API",
    description="MLOps API for multiclass market segment classification",
    version="1.0.0"
)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    """
    Loads the model from MLflow (Cloud Registry) or falls back to local artifact.
    """
    global model

    # Apuntar al servidor MLflow dentro de Kubernetes
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000"))

    try:
        # Intento 1: MLflow Model Registry de forma dinámica (Compatible con MLflow 2.10+)
        model_name = "HotelSegmentClassifier"
        client = MlflowClient()
        
        # Buscar todas las versiones del modelo registrado
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise Exception(f"No versions found for model {model_name}")
        
        # Obtener la versión numérica más alta (ej: versión 4)
        latest_version = max(int(v.version) for v in versions)
        model_uri = f"models:/{model_name}/{latest_version}"
        
        logger.info(f"Loading model from MLflow URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow.")
        
    except Exception as e:
        logger.warning(f"MLflow not reachable or model not found ({e}). Falling back to local DVC model.")
        try:
            # Intento 2: Archivo local (Producción o respaldo)
            model_path = os.getenv("MODEL_PATH", "models/model.joblib")
            model = joblib.load(model_path)
            logger.info("Model loaded successfully from local file.")
        except Exception as ex:
            logger.error(f"Critical error: Failed to load model locally: {ex}")


@app.post("/predict")
def predict_segment(features: BookingFeatures):
    """
    Predicts the market segment based on booking features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    try:
        # Convert Pydantic model to DataFrame for the scikit-learn pipeline
        input_data = pd.DataFrame([features.dict()])
        
        # Perform inference
        prediction = model.predict(input_data)
        
        return {
            "predicted_market_segment": str(prediction[0])
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))