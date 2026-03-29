import yaml, json, joblib
import pandas as pd
import mlflow, mlflow.xgboost
from xgboost import XGBClassifier
from pathlib import Path

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def train(params):
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    
    mlflow.set_experiment("fraud-detection")
    
    with mlflow.start_run():
        mlflow.log_params(params['xgboost'])
        
        model = XGBClassifier(**params['xgboost'], 
                             eval_metric='aucpr',
                             random_state=params['random_state'])
        model.fit(X_train, y_train)
        
        # Logger le modèle
        mlflow.xgboost.log_model(model, "model")
        
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/xgboost_fraud.pkl")
        joblib.dump(params['threshold'], "models/threshold.pkl")
        
        print("Training complete.")
        return model

if __name__ == "__main__":
    params = load_params()
    train(params)