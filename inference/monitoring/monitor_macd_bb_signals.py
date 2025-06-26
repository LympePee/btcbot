import os
import pandas as pd
from datetime import datetime, timedelta

from btcbot.utils.fetch_binance import fetch_ohlcv_binance

PREDICTIONS_CSV = "predictions/macd_bb_predictions.csv"
LABELED_CSV = "data/training_sets/macd_bb_labeled_predictions.csv"

SYMBOL = "BTC/USDC"
TIMEFRAME = "1h"
EVAL_HOURS = 4
THRESHOLD = 0.01  # 1% move = success

df = pd.read_csv(PREDICTIONS_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["evaluated"] = df.get("evaluated", False)
df["outcome"] = df.get("outcome", None)

now = datetime.utcnow()

for idx, row in df.iterrows():
    if row["evaluated"]:
        continue

    ts = row["timestamp"]
    if ts + timedelta(hours=EVAL_HOURS) > now:
        continue

    since = int(ts.timestamp() * 1000)
    ohlcv = fetch_ohlcv_binance(SYMBOL, TIMEFRAME, since=since)
    ohlcv_df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], unit="ms")
    future_df = ohlcv_df[ohlcv_df["timestamp"] > ts]

    if future_df.empty:
        continue

    entry_price = row["price"]
    future_close = future_df.iloc[-1]["close"]
    price_change = (future_close - entry_price) / entry_price

    df.at[idx, "outcome"] = int(price_change > THRESHOLD)
    df.at[idx, "evaluated"] = True

os.makedirs(os.path.dirname(LABELED_CSV), exist_ok=True)
df.to_csv(LABELED_CSV, index=False)
print(f"✅ Evaluated predictions saved to {LABELED_CSV}")
