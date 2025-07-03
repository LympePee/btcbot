# inference/monitoring/monitor_signals.py

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from btcbot.utils.fetch_binance import fetch_ohlcv_binance
from features.hull_features import apply_hull_features

# === Config ===
SYMBOL = "BTC/USDC"
TIMEFRAME = "1h"
DAYS_BACK = 2
MODEL_PATH = "models/hull_strategy_model.pkl"
SCALER_PATH = "models/hull_strategy_scaler.pkl"
OUTPUT_PATH = "predictions/hull_predictions.csv"

# === Load model and scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Fetch recent data ===
print("📡 Fetching recent data for BTC/USDC...")
since = int((datetime.utcnow() - timedelta(days=DAYS_BACK)).timestamp() * 1000)
df = fetch_ohlcv_binance(SYMBOL, TIMEFRAME, since)

# === Apply features ===
df = apply_hull_features(df)
df = df.dropna()

# === Keep latest row for prediction ===
latest = df.iloc[-1:]
feature_cols = [
    "hma_30m", "hma_1h", "hma_4h",
    "hma_30m_slope", "hma_1h_slope", "hma_4h_slope",
    "rsi", "cci", "volume"
]
latest_scaled = scaler.transform(latest[feature_cols])
prediction = model.predict(latest_scaled)[0]
confidence = model.predict_proba(latest_scaled)[0][prediction]

print(f"🤖 Hull Prediction: {prediction} | Confidence: {confidence:.2f}")

# === Save result ===
os.makedirs("predictions", exist_ok=True)
timestamp = latest["datetime"].values[0]
output_row = {
    "timestamp": timestamp,
    "symbol": SYMBOL,
    "prediction": int(prediction),
    "confidence": float(confidence),
    "entry_price": float(latest["close"].values[0])
}
out_df = pd.DataFrame([output_row])

# Append or create CSV
if os.path.exists(OUTPUT_PATH):
    out_df.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
else:
    out_df.to_csv(OUTPUT_PATH, index=False)

print(f"📄 Saved prediction to {OUTPUT_PATH}")
