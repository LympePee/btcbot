import pandas as pd
import joblib
import os

# === CONFIG ===
DATA_PATH = "data/training_sets/sl_training_data.csv"
MODEL_PATH = "models/stoploss_classifier.pkl"
SCALER_PATH = "models/stoploss_scaler.pkl"
OUTPUT_PATH = "predictions/stoploss_labeled_predictions.csv"

# === Load ===
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Predict ===
features = df.drop(columns=["timestamp", "drawdown_pct", "label_class"])
X_scaled = scaler.transform(features)
df["predicted_class"] = model.predict(X_scaled)

# === (Optional) Predict Probabilities ===
proba = model.predict_proba(X_scaled)
for i in range(proba.shape[1]):
    df[f"proba_{i}"] = proba[:, i]

# === Save ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved labeled predictions to {OUTPUT_PATH}")
