# backtest_macd_bb.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from btcbot.utils.fetch_binance import fetch_ohlcv_binance

def apply_macd_bb_features(df):
    # === MACD ===
    macd = ta.trend.MACD(df["close"])
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # === Bollinger Bands ===
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # === BB breakout % ===
    df["bb_breakout_pct"] = (df["close"] - df["bb_upper"]) / df["bb_upper"]
    df["bb_breakout"] = df["bb_breakout_pct"].apply(lambda x: 1 if x > 0.01 else 0)

    # === RSI ===
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    return df

def label_outcome(df):
    df["price_after_4h"] = df["close"].shift(-4)
    df["return_4h"] = (df["price_after_4h"] - df["close"]) / df["close"]
    df["outcome"] = df["return_4h"].apply(lambda x: 1 if x > 0.01 else 0)
    return df

def run_backtest(symbol="BTC/USDC", timeframe="1h", start_year=2017, end_year=datetime.now().year):
    final_df = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == datetime.now().year and month > datetime.now().month:
                break

            since = datetime(year, month, 1)
            since_ms = int(since.timestamp() * 1000)
            print(f"📅 Fetching {symbol} - {year}-{month:02d}")

            df = fetch_ohlcv_binance(symbol, timeframe, since_ms)
            if df.empty or len(df) < 100:
                print("⚠️ Skipping: Not enough data.")
                continue

            df = apply_macd_bb_features(df)
            df = label_outcome(df)
            df["symbol"] = symbol
            final_df = pd.concat([final_df, df], ignore_index=True)

    # Clean up
    final_df = final_df.dropna()
    final_df = final_df[final_df["bb_breakout"] == 1]  # Filter only breakouts
    final_df = final_df[final_df["return_4h"].abs() > 0.01]

    # Save
    os.makedirs("data/training_sets", exist_ok=True)
    final_df.to_csv("data/training_sets/macd_bb_filtered_signals.csv", index=False)
    print(f"✅ Done! Saved {len(final_df)} rows to macd_bb_filtered_signals.csv")

if __name__ == "__main__":
    run_backtest()
