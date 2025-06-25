# training/train_macd_bb.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# === Paths ===
DATA_PATH = "data/training_sets/macd_bb_filtered_signals.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "macd_bb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "macd_bb_scaler.pkl")

# === Load Data ===
df = pd.read_csv(DATA_PATH)

# === Features & Target ===
feature_cols = [
    "macd_line", "macd_signal",
    "bb_upper", "bb_lower",
    "bb_breakout_pct",
    "rsi"
]
target_col = "outcome"

X = df[feature_cols]
y = df[target_col]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Model Training ===
model = XGBClassifier(
    n_estimators=120,
    learning_rate=0.04,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train_scaled, y_train)

# === Save Model & Scaler ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# === Evaluation ===
accuracy = model.score(X_test_scaled, y_test)
print(f"✅ MACD+BB Model trained. Accuracy: {accuracy:.4f}")
print(f"📦 Saved model to: {MODEL_PATH}")
print(f"📦 Saved scaler to: {SCALER_PATH}")
