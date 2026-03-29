import json, joblib
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, f1_score)

def evaluate():
    model = joblib.load("models/xgboost_fraud.pkl")
    threshold = joblib.load("models/threshold.pkl")
    
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    
    metrics = {
        "auprc":     round(average_precision_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "threshold": threshold,
    }
    
    # Sauvegarder pour DVC metrics
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Logger dans MLflow (run actif)
    mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})
    mlflow.log_metric("threshold", metrics["threshold"])
    
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    evaluate()