# utils/fetch_kraken.py
import ccxt
import pandas as pd
import os
import time
from datetime import datetime

kraken = ccxt.kraken({'enableRateLimit': True})

def fetch_ohlcv_kraken(symbol: str, timeframe: str, since: int, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Kraken.
    """
    all_data = []
    while since < kraken.milliseconds():
        try:
            data = kraken.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not data:
                break
            all_data.extend(data)
            since = data[-1][0] + 1
            time.sleep(kraken.rateLimit / 1000)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def save_ohlcv_to_csv(df: pd.DataFrame, symbol: str, timeframe: str):
    symbol_clean = symbol.replace("/", "")
    folder = f"data/historical/kraken/{symbol_clean}/"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}{timeframe}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} rows to {filename}")

if __name__ == "__main__":
    symbol = "BTC/USDC"
    timeframe = "1h"
    since = kraken.parse8601("2024-01-01T00:00:00Z")

    df = fetch_ohlcv_kraken(symbol, timeframe, since)
    save_ohlcv_to_csv(df, symbol, timeframe)
