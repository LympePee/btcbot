#training/retrain_btc_dominance_model.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === Paths ===
DATA_PATH = "data/training_sets/BTC_Dominance_Training_Set.csv"
MODEL_PATH = "models/btc_dominance_classifier.pkl"
SCALER_PATH = "models/btc_dominance_scaler.pkl"

# === Load Data ===
df = pd.read_csv(DATA_PATH)
X = df[["btc_d_change", "btc_d_avg_change", "btc_d_volatility"]]
y = df["label"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === Evaluation ===
y_pred = model.predict(X_test_scaled)
print("🔁 Retraining BTC Dominance Model")
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save Model and Scaler ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
