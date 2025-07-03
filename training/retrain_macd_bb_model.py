# training/retrain_macd_bb_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# === Paths ===
LABELED_CSV = "data/training_sets/macd_bb_labeled_predictions.csv"
MODEL_PATH = "models/macd_bb_model.pkl"
SCALER_PATH = "models/macd_bb_scaler.pkl"

# === Load Labeled Data ===
df = pd.read_csv(LABELED_CSV)
df.dropna(inplace=True)

# === Features & Target ===
FEATURES = ["macd_line", "macd_signal", "bb_upper", "bb_lower", "rsi", "bb_breakout"]
TARGET = "outcome"

X = df[FEATURES]
y = df[TARGET]

# === Scale ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train Model ===
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X_scaled, y)

# === Save ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ MACD+BB Model retrained successfully.")
