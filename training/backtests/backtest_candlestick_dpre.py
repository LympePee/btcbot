import os
import json
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# === CONFIGURATION ===
symbol = "BTC/USDT"
exchange = ccxt.binance({'enableRateLimit': True})
start_date = "2017-01-01T00:00:00Z"
end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
output_path = "data/training_sets/candlestick_dpre_training_set.jsonl"

# === Robust Fetch Function ===
def fetch_ohlcv(tf, max_retry=12):
    print(f"📥 Fetching {tf} candles from {start_date} ...")
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    all_candles = []
    fail_count = 0

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        except Exception as e:
            print(f"⚠️ Error: {e}. Retrying in 10s...")
            fail_count += 1
            if fail_count > max_retry:
                print("❌ Too many fails. Aborting fetch.")
                break
            time.sleep(10)
            continue

        if not candles:
            print("❌ No more candles fetched (possibly hit end).")
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1
        fail_count = 0  # reset after a successful fetch

        if len(all_candles) % 5000 < 1000:
            print(f"   ...{len(all_candles)} candles so far ({tf})")
        time.sleep(0.5)  # Binance polite rate limit

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    return df

# === Fetch all timeframes safely ===
df_1h = fetch_ohlcv("1h")
df_4h = fetch_ohlcv("4h")
df_1d = fetch_ohlcv("1d")

# === Sync indexes (cut everything to start at max(20th_candle of each TF)) ===
if len(df_1h) < 24 or len(df_4h) < 24 or len(df_1d) < 24:
    raise ValueError("❌ Not enough data in one or more timeframes. Try expanding the date range.")

min_start = max(df_1h.index[20], df_4h.index[20], df_1d.index[20])
df_1h = df_1h[df_1h.index >= min_start]
df_4h = df_4h[df_4h.index >= min_start]
df_1d = df_1d[df_1d.index >= min_start]

# === Generate Labeled Sequences ===
samples = []
timestamps = df_1h.index[:-24]  # 20 past + 4 future

print("🔍 Generating labeled sequences...")

for idx, ts in enumerate(timestamps):
    try:
        past_1h = df_1h.loc[ts - pd.Timedelta(hours=19):ts]
        past_4h = df_4h.loc[ts - pd.Timedelta(hours=76):ts]
        past_1d = df_1d.loc[ts - pd.Timedelta(days=19):ts]
        future = df_1h.loc[ts + pd.Timedelta(hours=1):ts + pd.Timedelta(hours=4)]

        if len(past_1h) < 20 or len(past_4h) < 20 or len(past_1d) < 20 or len(future) < 4:
            continue

        current_close = df_1h.loc[ts]["close"]
        future_max = future["high"].max()
        future_min = future["low"].min()
        last_future_close = future["close"].iloc[-1]

        # === Label logic ===
        if (future_max - current_close) / current_close > 0.01:
            label = "bullish"
        elif (current_close - future_min) / current_close > 0.01:
            label = "bearish"
        elif abs((last_future_close - current_close) / current_close) < 0.005:
            label = "sideways"
        else:
            continue  # Skip ambiguous

        sample = {
            "1h_sequence": past_1h[["open", "high", "low", "close", "volume"]].values.tolist(),
            "4h_sequence": past_4h[["open", "high", "low", "close", "volume"]].values.tolist(),
            "1d_sequence": past_1d[["open", "high", "low", "close", "volume"]].values.tolist(),
            "target": label
        }
        samples.append(sample)

        if idx % 5000 == 0 and idx > 0:
            print(f"  ...Processed {idx} 1h steps, {len(samples)} valid samples.")

    except Exception as e:
        print(f"⚠️ Skipped sample at {ts} due to: {e}")
        continue

# === SAVE TO JSONL ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for item in samples:
        f.write(json.dumps(item) + "\n")

print(f"✅ Saved {len(samples)} samples to {output_path}")
