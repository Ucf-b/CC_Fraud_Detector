# monitoring/monitor.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import json
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Charger le modèle
model = XGBClassifier()
model.load_model("models/xgboost_fraud_best.json")
with open("models/threshold_best.json") as f:
    threshold = json.load(f)["threshold"]

# Charger les données
df = pd.read_csv("data/creditcard.csv")
df['LogAmount'] = np.log1p(df['Amount'])

features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['LogAmount']
X = df[features]
y = df['Class']

# Split référence / production
from sklearn.model_selection import train_test_split
X_ref, X_prod, y_ref, y_prod = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Simuler du data drift sur la production
X_prod_drifted = X_prod.copy()
X_prod_drifted['V14'] += np.random.normal(0, 0.5, len(X_prod))
X_prod_drifted['V4']  += np.random.normal(0, 0.3, len(X_prod))

# Ajouter les prédictions
X_ref['prediction']  = (model.predict_proba(X_ref)[:, 1] >= threshold).astype(int)
X_ref['target']      = y_ref.values

X_prod_drifted['prediction'] = (model.predict_proba(X_prod_drifted)[:, 1] >= threshold).astype(int)
X_prod_drifted['target']     = y_prod.values

# Générer le rapport
Path("reports").mkdir(exist_ok=True)
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_ref, current_data=X_prod_drifted)
report.save_html("reports/drift_report.html")

print("✅ Rapport généré → reports/drift_report.html")

# Détecter le drift et alerter
report_dict = report.as_dict()
drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
n_drifted = report_dict['metrics'][0]['result']['number_of_drifted_columns']

if drift_detected:
    print(f"⚠️  DRIFT DÉTECTÉ — {n_drifted} features ont drifté !")
else:
    print(f"✅ Pas de drift détecté")