import os
import json
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# === CONFIGURATION ===
symbol = "BTC/USDT"
exchange = ccxt.binance()
start_date = "2017-01-01T00:00:00Z"  # πάμε πίσω στο 2017
end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
output_path = "data/training_sets/candlestick_dpre_training_set.jsonl"

# === FETCH FUNCTION (διορθωμένο) ===
def fetch_ohlcv(tf):
    print(f"📥 Fetching {tf} candles...")
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    all_candles = []

    while since < end_ts:
        candles = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    return df

# === Fetch All Timeframes ===
df_1h = fetch_ohlcv("1h")
df_4h = fetch_ohlcv("4h")
df_1d = fetch_ohlcv("1d")

# === Sync indexes σωστά ===
common_start = max(df_1h.index[20], df_4h.index[20], df_1d.index[20])
df_1h = df_1h[df_1h.index >= common_start]
df_4h = df_4h[df_4h.index >= common_start]
df_1d = df_1d[df_1d.index >= common_start]

# === ALIGN AND GENERATE SAMPLES ===
samples = []
timestamps = df_1h.index[20:-4]  # χρειάζεσαι 20 past + 4 future candles

print("🔍 Generating labeled sequences...")
for ts in timestamps:
    try:
        past_1h = df_1h.loc[ts - pd.Timedelta(hours=19):ts]
        past_4h = df_4h.loc[:ts].iloc[-20:]
        past_1d = df_1d.loc[:ts].iloc[-20:]
        future = df_1h.loc[ts + pd.Timedelta(hours=1):ts + pd.Timedelta(hours=4)]

        if len(past_1h) < 20 or len(past_4h) < 20 or len(past_1d) < 20 or len(future) < 4:
            continue

        current_close = df_1h.loc[ts]["close"]
        future_max = future["high"].max()
        future_min = future["low"].min()

        # Καθορισμός labels ξεκάθαρα
        if (future_max - current_close) / current_close > 0.01:
            label = "bullish"
        elif (current_close - future_min) / current_close > 0.01:
            label = "bearish"
        elif abs((future["close"].iloc[-1] - current_close) / current_close) <= 0.005:
            label = "sideways"
        else:
            continue  # αποφεύγεις θόρυβο

        sample = {
            "timestamp": ts.isoformat(),
            "1h_sequence": past_1h[["open", "high", "low", "close", "volume"]].values.tolist(),
            "4h_sequence": past_4h[["open", "high", "low", "close", "volume"]].values.tolist(),
            "1d_sequence": past_1d[["open", "high", "low", "close", "volume"]].values.tolist(),
            "target": label
        }
        samples.append(sample)
    except Exception as e:
        print(f"Skipped sample at {ts} due to error: {e}")
        continue

# === SAVE ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for item in samples:
        f.write(json.dumps(item) + "\n")

print(f"✅ Saved {len(samples)} samples to {output_path}")
