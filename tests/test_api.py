from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

SAMPLE_TRANSACTION = {
    "Time": 406.0,
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
    "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.18, "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.19, "V27": 0.03, "V28": 0.01,
    "Amount": 143.72,
    "LogAmount": 4.97
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data
    assert "auprc" in data

def test_predict_returns_valid_response():
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= data["fraud_probability"] <= 1.0

def test_predict_missing_field():
    bad_transaction = SAMPLE_TRANSACTION.copy()
    del bad_transaction["V1"]
    response = client.post("/predict", json=bad_transaction)
    assert response.status_code == 422  # Unprocessable Entity