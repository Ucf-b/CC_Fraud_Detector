# 🔍 Credit Card Fraud Detection — MLOps Pipeline

![CI](https://github.com/Ucf-b/CC_Fraud_Detector/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![MLflow](https://img.shields.io/badge/tracking-MLflow%20%2F%20DagsHub-orange)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end MLOps pipeline for real-time credit card fraud detection on an extremely imbalanced dataset (0.17% fraud rate). From exploratory analysis to production serving with monitoring.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| AUPRC | **0.88** |
| Precision | **0.94** |
| Recall | **0.81** |
| Top 0.5% capture | **89.8% of frauds** |

> Investigating only **0.5% of transactions** captures nearly **90% of all fraud cases**.

---

## 🏗️ Architecture

```
Raw Data (284k transactions)
        ↓
  Feature Engineering
  (log_amount, V1-V28, Time)
        ↓
  XGBoost Training
  (tracked with MLflow on DagsHub)
        ↓
  Threshold Optimization
  (precision-recall curve)
        ↓
  FastAPI REST API ──────────── Docker Container
        ↓
  GitHub Actions CI/CD
        ↓
  Evidently AI Monitoring
  (data drift detection)
```

---

## 🚀 Quick Start

### Run with Docker

```bash
docker pull ucfb/fraud-api
docker run -p 8000:8000 ucfb/fraud-api
```

### Run locally

```bash
git clone https://github.com/Ucf-b/CC_Fraud_Detector
cd CC_Fraud_Detector
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

API available at **http://localhost:8000/docs**

---

## 📡 API Endpoints

### `POST /predict`

Classify a transaction as fraudulent or legitimate.

**Request body:**
```json
{
  "Time": 406.0,
  "V1": -1.35, "V2": -0.07, "...", "V28": 0.01,
  "LogAmount": 4.97
}
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "risk_level": "LOW",
  "threshold_used": 0.42
}
```

| `risk_level` | Meaning | Recommended action |
|---|---|---|
| `LOW` | proba < 0.5 | Auto-approve |
| `MEDIUM` | 0.5 ≤ proba < 0.8 | Flag for review |
| `HIGH` | proba ≥ 0.8 | Block + alert |

### `GET /health`
```json
{ "status": "ok", "model": "xgboost-fraud-v1" }
```

### `GET /model-info`
```json
{
  "model_type": "XGBoost",
  "features": ["Time", "V1-V28", "LogAmount"],
  "threshold": 0.42,
  "auprc": 0.88,
  "precision": 0.94,
  "recall": 0.81
}
```

---

## 📁 Project Structure

```
CC_Fraud_Detector/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── models/
│   ├── xgboost_fraud_best.json # Trained model (native XGBoost format)
│   └── threshold_best.json     # Optimized classification threshold
├── monitoring/
│   └── monitor.py              # Evidently AI drift detection
├── reports/
│   └── drift_report.html       # Generated monitoring report
├── src/
│   ├── api/
│   │   └── main.py             # FastAPI application
│   └── models/
│       └── predict.py          # Prediction utilities
├── tests/
│   └── test_api.py             # pytest test suite
├── conftest.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧪 Model Details

### Dataset
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud rate:** 0.17% (highly imbalanced)
- **Features:** Time, V1–V28 (PCA), Amount → log-transformed to LogAmount

### Feature Engineering
- `LogAmount = log(Amount + 1)` — reduces skew from outliers above 25,000
- PCA features V1–V28 retained (V14, V4, V12 most predictive per SHAP)

### Models Compared

| Model | AUPRC |
|-------|-------|
| Random Forest | 0.85 |
| LightGBM | 0.85 |
| XGBoost | 0.87 |
| **XGBoost (tuned)** | **0.88** |

### Threshold Optimization
Default threshold (0.5) replaced by optimal threshold from precision-recall curve, maximizing F1 score on the test set.

### Top-K Fraud Capture

| Flagged transactions | Frauds captured |
|---|---|
| Top 0.1% | 57.1% |
| Top 0.5% | 89.8% |
| Top 1% | 90.8% |
| Top 2% | 91.8% |

### Key Features (SHAP)
`V14` > `V4` > `V12` — extreme values in these PCA components strongly increase fraud probability.

---

## 🔬 Experiment Tracking

All 6 training runs are tracked on **DagsHub / MLflow**:

👉 [View experiments on DagsHub](https://dagshub.com/Ucf-b/CC_Fraud_Detector.mlflow)

Tracked per run: hyperparameters, AUPRC, precision, recall, F1, threshold, model artifact.

---

## 📈 Monitoring

Run the drift detection report locally:

```bash
python monitoring/monitor.py
```

Opens `reports/drift_report.html` — an interactive Evidently report comparing:
- Feature distributions (reference vs production)
- Data drift score per feature
- Alert if drift detected on key features (V14, V4, V12)

**Why this matters:** fraud patterns evolve over time. Monitoring detects model degradation before it impacts business metrics.

---

## ⚙️ CI/CD

Every push to `main` triggers:

1. **Lint check** — code quality
2. **pytest** — 4 automated tests (health, model-info, predict, validation)
3. **Docker build** — ensures the container builds successfully

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost, Scikit-learn |
| Experiment tracking | MLflow, DagsHub |
| Serving | FastAPI, Pydantic, Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |
| Testing | pytest, httpx |
| Data | Pandas, NumPy |

---

## 👤 Author

**Youssef IBN SEDDIK** — CentraleSupélec  
[LinkedIn](https://www.linkedin.com/in/youssef-ibn-seddik) · [GitHub](https://github.com/Ucf-b)
