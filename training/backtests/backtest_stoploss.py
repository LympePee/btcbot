import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator

# === PATH CONFIG ===
CSV_PATH = "data/historical/binance/BTCUSDC/1h.csv"
OUTPUT_PATH = "data/training_sets/sl_training_data.csv"
MIN_DRAWDOWN_PCT = 1.0
MAX_LOOKAHEAD_HOURS = 6

# === FEATURE EXTRACTION ===
def extract_stoploss_features(df: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Indicators
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14, fillna=True).average_true_range()
    df["ema_20"] = EMAIndicator(df["close"], window=20, fillna=True).ema_indicator()
    df["ema_50"] = EMAIndicator(df["close"], window=50, fillna=True).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14, fillna=True).rsi()
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd_hist"] = macd.macd_diff()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3, fillna=True)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["volume_spike"] = (df["volume"] / df["volume_sma_20"]).clip(0.5, 10)

    try:
        row = df.loc[current_time]
    except KeyError:
        return None

    current_price = row["close"]
    candle_size = abs(row["close"] - row["open"])
    candle_class = (
        "bull" if row["close"] > row["open"]
        else "bear" if row["close"] < row["open"]
        else "neutral"
    )

    return {
        "timestamp": int(current_time.timestamp() * 1000),  # keep ms format
        "atr": row["atr"],
        "atr_ratio": row["atr"] / current_price,
        "ema_distance_pct": ((current_price - row["ema_20"]) / row["ema_20"]) * 100,
        "price_vs_ema50": current_price / row["ema_50"],
        "rsi": row["rsi"],
        "macd_hist": row["macd_hist"],
        "stoch_k": row["stoch_k"],
        "stoch_d": row["stoch_d"],
        "volume_spike": row["volume_spike"],
        "candle_class_bull": int(candle_class == "bull"),
        "candle_class_bear": int(candle_class == "bear"),
        "candle_class_neutral": int(candle_class == "neutral"),
        "candle_size_ratio": candle_size / current_price,
    }

# === DRAWNDOWN CLASSIFICATION ===
def classify_drawdown(pct):
    if pct < 1.5:
        return 0
    elif pct < 2.5:
        return 1
    elif pct < 4:
        return 2
    elif pct < 6:
        return 3
    else:
        return 4

# === DRAWDOWN DETECTION ===
def detect_drops_and_generate_rows(df: pd.DataFrame):
    rows = []
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    for i in range(14, len(df) - MAX_LOOKAHEAD_HOURS):
        start_time = df.index[i]
        start_price = df["high"].iloc[i]
        future_lows = df["low"].iloc[i+1:i+MAX_LOOKAHEAD_HOURS+1]
        min_low = future_lows.min()
        drawdown_pct = ((start_price - min_low) / start_price) * 100

        if drawdown_pct >= MIN_DRAWDOWN_PCT:
            features = extract_stoploss_features(df.reset_index(), start_time)
            if features:
                features["drawdown_pct"] = drawdown_pct
                features["label_class"] = classify_drawdown(drawdown_pct)
                rows.append(features)

    return pd.DataFrame(rows)

# === MAIN MONTHLY LOOP ===
df_full = pd.read_csv(CSV_PATH)
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], unit="ms")
start = df_full["timestamp"].min().replace(day=1, hour=0, minute=0)
end = df_full["timestamp"].max()
current = start

final_df = pd.DataFrame()

while current <= end:
    next_month = current + relativedelta(months=1)
    df_month = df_full[(df_full["timestamp"] >= current) & (df_full["timestamp"] < next_month)].copy()

    if not df_month.empty:
        print(f"📆 Processing {current.strftime('%Y-%m')}")
        result = detect_drops_and_generate_rows(df_month)
        final_df = pd.concat([final_df, result], ignore_index=True)

    current = next_month

# Save final training set
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(final_df)} rows to {OUTPUT_PATH}")
