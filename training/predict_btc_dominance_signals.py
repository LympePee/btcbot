# btcbot/training/predict_btc_dominance_signals.py

import os
import pandas as pd
import joblib
import ccxt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
symbol = "BTC/USDT"
timeframe = "1h"
exchange = ccxt.binance()
lookback_hours = 48
model_path = "models/btc_dominance_classifier.pkl"
scaler_path = "models/btc_dominance_scaler.pkl"
output_path = "data/training_sets/btc_dominance_labeled_predictions.csv"

# === Load model and scaler ===
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === Fetch recent data ===
now = datetime.utcnow()
since = int((now - timedelta(hours=lookback_hours)).timestamp() * 1000)
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)

# === Convert to DataFrame ===
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("datetime", inplace=True)

# === Calculate features ===
df["d_change"] = df["close"].pct_change() * 100
df["btc_d_avg_change"] = df["d_change"].rolling(window=24).mean()
df["btc_d_volatility"] = df["d_change"].rolling(window=24).std()
df["btc_d_change"] = df["d_change"]

# === Drop NaNs and prepare input ===
df.dropna(inplace=True)
features = df[["btc_d_change", "btc_d_avg_change", "btc_d_volatility"]].copy()
scaled = scaler.transform(features)
df["label"] = model.predict(scaled)

# === Export predictions ===
df = df[["timestamp", "btc_d_change", "btc_d_avg_change", "btc_d_volatility", "label"]]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Saved {len(df)} rows to {output_path}")
