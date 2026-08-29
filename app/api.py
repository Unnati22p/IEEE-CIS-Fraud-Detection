from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import joblib
import numpy as np
import pandas as pd
import os


# ============================================================
# IEEE-CIS FRAUD DETECTION API
# XGBoost Model
# ============================================================

app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="API for predicting potentially fraudulent online transactions.",
    version="1.0.0"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# Optimized classification threshold obtained during evaluation
FRAUD_THRESHOLD = 0.8294


# ============================================================
# LOAD SAVED MODEL OBJECTS
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    # Encoder is loaded for consistency with the training pipeline.
    # The current API expects already-preprocessed numerical features.
    encoder = joblib.load(ENCODER_PATH)

    print("Model loaded successfully.")
    print("Number of expected features:", len(feature_columns))

except Exception as e:
    model = None
    feature_columns = None
    encoder = None

    print("Model files are not available yet.")
    print("Expected files:")
    print(MODEL_PATH)
    print(FEATURES_PATH)
    print(ENCODER_PATH)
    print("Error:", e)


# ============================================================
# REQUEST FORMAT
# ============================================================

class PredictionRequest(BaseModel):
    features: Dict[str, float]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "IEEE-CIS Fraud Detection API",
        "status": "running",
        "model": "XGBoost",
        "fraud_threshold": FRAUD_THRESHOLD
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    if model is None:
        return {
            "status": "running",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(feature_columns)
    }


# ============================================================
# FRAUD PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model files are not available. Place fraud_model.pkl, features.pkl and encoder.pkl inside the models folder."
        )

    if feature_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Feature configuration could not be loaded."
        )

    # --------------------------------------------------------
    # Check for missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in request.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features
            }
        )

    # --------------------------------------------------------
    # Create dataframe in the exact training feature order
    # --------------------------------------------------------

    input_data = {
        feature: request.features[feature]
        for feature in feature_columns
    }

    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Make sure all values are numeric
    # --------------------------------------------------------

    try:
        input_df = input_df.astype(float)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"All feature values must be numerical. Error: {str(e)}"
        )

    # --------------------------------------------------------
    # Generate fraud probability
    # --------------------------------------------------------

    try:
        fraud_probability = float(
            model.predict_proba(input_df)[0][1]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Apply optimized threshold
    # --------------------------------------------------------

    prediction = int(
        fraud_probability >= FRAUD_THRESHOLD
    )

    if prediction == 1:
        prediction_label = "Fraud"
    else:
        prediction_label = "Not Fraud"

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "fraud_probability": round(fraud_probability, 6),
        "threshold": FRAUD_THRESHOLD
    }


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
