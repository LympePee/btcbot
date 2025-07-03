import pandas as pd
import ccxt
from datetime import datetime, timedelta
import os

# === CONFIG ===
symbol = "BTC/USDT"
exchange = ccxt.binance()
timeframe = "1h"
start_date = "2022-01-01T00:00:00Z"
end_date = "2024-01-01T00:00:00Z"
output_path = "data/training_sets/BTC_Dominance_Training_Set.csv"

# === Fetch OHLCV ===
print("⏳ Downloading data...")
since = exchange.parse8601(start_date)
end_ts = exchange.parse8601(end_date)

all_data = []
while since < end_ts:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    if not ohlcv:
        break
    all_data.extend(ohlcv)
    since = ohlcv[-1][0] + 1

# === DataFrame Conversion ===
df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("datetime", inplace=True)

# === Features ===
df["d_change"] = df["close"].pct_change() * 100
df["btc_d_avg_change"] = df["d_change"].rolling(window=24).mean()
df["btc_d_volatility"] = df["d_change"].rolling(window=24).std()
df["btc_d_change"] = df["d_change"]

# === Label ===
future_returns = df["close"].shift(-6) / df["close"] - 1
df["label"] = (future_returns > 0.003).astype(int)

# === Clean & Save ===
df = df[["btc_d_change", "btc_d_avg_change", "btc_d_volatility", "label"]].dropna()
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Saved {len(df)} rows to {output_path}")
