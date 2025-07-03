import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator


def extract_stoploss_features(ohlcv_df: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    """
    Υπολογίζει τα χαρακτηριστικά (features) για ML stop-loss εκτίμηση, βασισμένο σε στιγμιότυπο αγοράς πριν από πτώση.

    Parameters:
        ohlcv_df: DataFrame με στήλες ["timestamp", "open", "high", "low", "close", "volume"]
        current_time: Το κερί που θέλουμε να αξιολογήσουμε (timestamp)

    Returns:
        dict με τα τεχνικά χαρακτηριστικά (features)
    """
    df = ohlcv_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # --- Τεχνικοί δείκτες ---
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

    # Candle τύπος & μέγεθος
    try:
        row = df.loc[current_time]
    except KeyError:
        raise ValueError(f"Timestamp {current_time} not found in OHLCV data.")

    current_price = row["close"]
    candle_size = abs(row["close"] - row["open"])
    candle_class = (
        "bull" if row["close"] > row["open"]
        else "bear" if row["close"] < row["open"]
        else "neutral"
    )

    # Feature dictionary
    features = {
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

    return features
