# training/retrain_hull_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# === Paths ===
DATA_PATH = "data/training_sets/hull_filtered_signals.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hull_strategy_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "hull_strategy_scaler.pkl")

# === Load Data ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Retraining dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# === Features & Target ===
feature_cols = [
    "hma_30m", "hma_1h", "hma_4h",
    "hma_30m_slope", "hma_1h_slope", "hma_4h_slope",
    "rsi", "cci", "volume"
]
target_col = "outcome"

if not all(col in df.columns for col in feature_cols + [target_col]):
    raise ValueError("❌ Missing required columns in dataset.")

X = df[feature_cols]
y = df[target_col]

# === Scale ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Retrain ===
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_scaled, y)

# === Save updated model & scaler ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Hull Strategy Model retrained and saved.")
