import os
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import numpy as np

# === Ρυθμίσεις ===
PREDICTIONS_CSV = "predictions/hull_predictions.csv"
LABELED_CSV = "data/training_sets/hull_labeled_predictions.csv"
SYMBOL = "BTC/USDC"
TIMEFRAME = "1h"
EVAL_HOURS = 4   # Πόσες ώρες μετά να κοιτάμε την τιμή;

# === Φόρτωσε τις προβλέψεις ===
df = pd.read_csv(PREDICTIONS_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
if "outcome" not in df.columns:
    df["outcome"] = None

# === Exchange ===
exchange = ccxt.binance()

now = datetime.utcnow()

for idx, row in df.iterrows():
    # Skip αν υπάρχει ήδη outcome
    if pd.notna(row["outcome"]):
        continue

    ts = row["timestamp"]
    entry_price = row.get("price", None) or row.get("entry_price", None) or row.get("close", None)
    if pd.isna(ts) or pd.isna(entry_price):
        continue

    # Θέλουμε να έχουν περάσει 4 ώρες για αξιολόγηση
    if ts + timedelta(hours=EVAL_HOURS) > now:
        continue

    since = int(ts.timestamp() * 1000)
    future_ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since)
    ohlcv_df = pd.DataFrame(future_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], unit="ms")

    # Βρες το close 4 ώρες μετά
    future_candle = ohlcv_df[ohlcv_df["timestamp"] > ts]
    if future_candle.shape[0] < EVAL_HOURS:
        continue  # δεν υπάρχουν αρκετά κεριά στο μέλλον
    price_after_4h = future_candle.iloc[EVAL_HOURS - 1]["close"]

    if pd.isna(price_after_4h):
        continue

    # Υπολόγισε scaled outcome: (price_after_4h - entry) / entry, clamp -1 .. 1
    price_change = (price_after_4h - entry_price) / entry_price
    outcome_score = max(-1, min(1, price_change))
    df.at[idx, "outcome"] = round(outcome_score, 4)

# Αποθήκευσε!
os.makedirs(os.path.dirname(LABELED_CSV), exist_ok=True)
df.to_csv(LABELED_CSV, index=False)
print(f"✅ Labeled predictions saved to {LABELED_CSV}")
