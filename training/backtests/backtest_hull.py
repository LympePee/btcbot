# training/backtests/backtest_hull.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import CCIIndicator
import ta.momentum
from btcbot.utils.fetch_binance import fetch_ohlcv_binance

def calculate_hma(series, period):
    half = int(period / 2)
    sqrt = int(np.sqrt(period))

    wma_half = series.rolling(window=half).mean()
    wma_full = series.rolling(window=period).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=sqrt).mean()
    return hma

def apply_hull_features(df):
    df["hma_30m"] = calculate_hma(df["close"], 9)
    df["hma_1h"] = calculate_hma(df["close"], 16)
    df["hma_4h"] = calculate_hma(df["close"], 25)

    df["hma_30m_slope"] = df["hma_30m"].diff()
    df["hma_1h_slope"] = df["hma_1h"].diff()
    df["hma_4h_slope"] = df["hma_4h"].diff()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

    return df

def label_outcome(df):
    df["price_after_4h"] = df["close"].shift(-4)
    df["return_4h"] = (df["price_after_4h"] - df["close"]) / df["close"]
    df["outcome"] = df["return_4h"].apply(lambda x: 1 if x > 0.01 else 0)
    return df

def run_backtest(symbol="BTC/USDC", timeframe="1h", start_year=2017):
    final_df = pd.DataFrame()
    today = datetime.today()
    current_year = today.year
    current_month = today.month

    for year in range(start_year, current_year + 1):
        for month in range(1, 13):
            if year == current_year and month > current_month:
                break  # μην πας σε μελλοντικό μήνα

            since = datetime(year, month, 1)
            since_ms = int(since.timestamp() * 1000)
            print(f"📅 Fetching {symbol} - {year}-{month:02d}")

            df = fetch_ohlcv_binance(symbol, timeframe, since_ms)
            if df.empty or len(df) < 100:
                print("⚠️ Skipping: Not enough data.")
                continue

            df = apply_hull_features(df)
            df = label_outcome(df)
            df["symbol"] = symbol
            final_df = pd.concat([final_df, df], ignore_index=True)

    # Clean up
    final_df = final_df.dropna()
    final_df = final_df[final_df["return_4h"].abs() > 0.01]

    # Save
    os.makedirs("data/training_sets", exist_ok=True)
    final_df.to_csv("data/training_sets/hull_filtered_signals.csv", index=False)
    print(f"✅ Done! Saved {len(final_df)} rows to hull_filtered_signals.csv")

if __name__ == "__main__":
    run_backtest()
