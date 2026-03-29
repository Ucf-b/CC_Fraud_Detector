from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import uvicorn
from pathlib import Path
import pandas as pd

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Détecte les transactions frauduleuses via XGBoost",
    version="1.0.0"
)

# Charger le modèle au démarrage
MODEL_PATH = Path("models/xgboost_fraud_best.pkl")
THRESHOLD_PATH = Path("models/threshold_best.pkl")

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

class Transaction(BaseModel):
    Time: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    LogAmount: float = Field(..., ge=0, description="log(Amount + 1)")

class Prediction(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str        # LOW / MEDIUM / HIGH
    threshold_used: float

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost-fraud-v1"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "XGBoost",
        "features": ["Time", "V1-V28", "LogAmount"],  # 30 features au total
        "threshold": threshold,
        "auprc": 0.88,
        "precision": 0.94,
        "recall": 0.81,
    }

@app.post("/predict", response_model=Prediction)
@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    features = pd.DataFrame([[
        transaction.Time,
        transaction.V1,  transaction.V2,  transaction.V3,  transaction.V4,
        transaction.V5,  transaction.V6,  transaction.V7,  transaction.V8,
        transaction.V9,  transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.LogAmount
    ]], columns=model.feature_names_in_)  # ← noms exacts du modèle
    
    proba = float(model.predict_proba(features)[0, 1])
    is_fraud = proba >= threshold
    
    if proba >= 0.8:
        risk = "HIGH"
    elif proba >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    return Prediction(
        is_fraud=is_fraud,
        fraud_probability=round(proba, 4),
        risk_level=risk,
        threshold_used=round(threshold, 4)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)