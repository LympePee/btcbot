import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# === Paths ===
HULL_CSV = "data/training_sets/hull_labeled_predictions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "meta_decision_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "meta_decision_scaler.pkl")

# === Load predictions ===
df_hull = pd.read_csv(HULL_CSV, parse_dates=["timestamp"])

# --- Ελέγχει για outcome
if "outcome" not in df_hull.columns:
    raise ValueError("❌ Δεν υπάρχει outcome στο Hull predictions.")

# === Features & target
feature_cols = ["prediction", "confidence"]
target_col = "outcome"

X = df_hull[feature_cols]
y = df_hull[target_col]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Model Training ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# === Save Model & Scaler ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ Meta-Decision Model trained and saved to {MODEL_PATH}")
acc = clf.score(X_test_scaled, y_test)
print(f"📊 Test accuracy: {acc:.4f}")
