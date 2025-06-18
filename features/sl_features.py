import pandas as pd
import numpy as np


def calculate_dynamic_sl_tp(df: pd.DataFrame, atr_window=14, tp_multiplier=2.0, sl_multiplier=1.0):
    """
    Υπολογίζει δυναμικά Take-Profit και Stop-Loss επίπεδα με βάση την ATR.
    """
    df = df.copy()

    # ATR
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = np.abs(df["high"] - df["close"].shift())
    df["low_close"] = np.abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=atr_window).mean()

    # Dynamic thresholds
    df["dynamic_tp"] = df["close"] + tp_multiplier * df["atr"]
    df["dynamic_sl"] = df["close"] - sl_multiplier * df["atr"]

    # Optional: ATR % of price
    df["atr_percent"] = df["atr"] / df["close"] * 100

    return df[["timestamp", "close", "atr", "atr_percent", "dynamic_tp", "dynamic_sl"]]


def generate_sl_features(timeframes_data: dict):
    """
    Δημιουργεί χαρακτηριστικά SL/TP για κάθε timeframe: 30m, 1h, 4h
    """
    sl_features = {}

    for tf, df in timeframes_data.items():
        df_sl = calculate_dynamic_sl_tp(df)
        sl_features[f"atr_{tf}"] = df_sl["atr"].values[-1]
        sl_features[f"atr_pct_{tf}"] = df_sl["atr_percent"].values[-1]
        sl_features[f"tp_{tf}"] = df_sl["dynamic_tp"].values[-1]
        sl_features[f"sl_{tf}"] = df_sl["dynamic_sl"].values[-1]

    return sl_features


# Παράδειγμα χρήσης για CSV ή real-time δεδομένα
if __name__ == "__main__":
    dfs = {
        "30m": pd.read_csv("data/historical/binance/BTCUSDC/30m.csv"),
        "1h": pd.read_csv("data/historical/binance/BTCUSDC/1h.csv"),
        "4h": pd.read_csv("data/historical/binance/BTCUSDC/4h.csv"),
    }
    features = generate_sl_features(dfs)
    print("📉 SL Features:", features)
