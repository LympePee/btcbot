# inference/predict_macd_bb.py

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import joblib
import ccxt
import ta
import numpy as np

# === Settings ===
SYMBOL = "BTC/USDC"
TIMEFRAME = "1h"
LOOKBACK_HOURS = 48
PREDICTIONS_CSV = "predictions/macd_bb_predictions.csv"

# === Load Model & Scaler ===
model = joblib.load("models/macd_bb_model.pkl")
scaler = joblib.load("models/macd_bb_scaler.pkl")

# === Fetch Data ===
exchange = ccxt.binance()
since = int((datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).timestamp() * 1000)
ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since)
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# === Compute Indicators ===
df["macd_line"] = ta.trend.macd(df["close"], fillna=True)
df["macd_signal"] = ta.trend.macd_signal(df["close"], fillna=True)

bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2, fillna=True)
df["bb_upper"] = bb.bollinger_hband()
df["bb_lower"] = bb.bollinger_lband()

df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14, fillna=True).rsi()
df["bb_breakout"] = (df["close"] - df["bb_upper"]) / df["close"]

# === Use last row for prediction ===
latest = df.iloc[-1]
features = [
    latest["macd_line"],
    latest["macd_signal"],
    latest["bb_upper"],
    latest["bb_lower"],
    latest["rsi"],
    latest["bb_breakout"]
]
X = scaler.transform([features])
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0][1]

# === Save Prediction ===
os.makedirs("predictions", exist_ok=True)
prediction_row = {
    "timestamp": latest["timestamp"],
    "symbol": SYMBOL,
    "macd_line": features[0],
    "macd_signal": features[1],
    "bb_upper": features[2],
    "bb_lower": features[3],
    "rsi": features[4],
    "bb_breakout": features[5],
    "prediction": int(pred),
    "confidence": round(proba, 4)
}

# Append to CSV
if os.path.exists(PREDICTIONS_CSV):
    pd.DataFrame([prediction_row]).to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)
else:
    pd.DataFrame([prediction_row]).to_csv(PREDICTIONS_CSV, index=False)

# === Print ===
print(f"🤖 MACD+BB Prediction: {pred} | Confidence: {proba:.4f}")
print(f"📄 Saved to: {PREDICTIONS_CSV}")
