# training/train_stoploss_classifier.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# === CONFIG ===
DATA_PATH = "data/training_sets/sl_training_data.csv"
MODEL_PATH = "btcbot/models/stoploss_classifier.pkl"
SCALER_PATH = "btcbot/models/stoploss_scaler.pkl"

# === 1. Load Data ===
df = pd.read_csv(DATA_PATH)

# === 2. Feature Selection ===
label_col = "label_class"
exclude = ["timestamp", "drawdown_pct", label_col]
X = df.drop(columns=exclude)
y = df[label_col]

# === 3. Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Train Model ===
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# === 6. Evaluate ===
y_pred = model.predict(X_test)
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 7. Save Model & Scaler ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
