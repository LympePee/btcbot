# inference/predict_hull.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import load
from btcbot.utils.fetch_binance import fetch_ohlcv_binance
from ta.trend import CCIIndicator
from ta.momentum import RSIIndicator

# === Load model and scaler ===
model = load("models/hull_strategy_model.pkl")
scaler = load("models/hull_strategy_scaler.pkl")

# === Hull feature logic ===
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

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

    return df

# === Predict Logic ===
def predict_latest(df):
    df = df.copy()
    df = apply_hull_features(df)
    df = df.dropna().copy()

    # extract last row for inference
    latest_row = df[[
        "hma_30m", "hma_1h", "hma_4h",
        "hma_30m_slope", "hma_1h_slope", "hma_4h_slope",
        "rsi", "cci", "volume"
    ]].iloc[[-1]]

    latest_scaled = scaler.transform(latest_row)
    pred = model.predict(latest_scaled)[0]
    prob = model.predict_proba(latest_scaled)[0][1]
    return pred, prob

# === Run Prediction ===
if __name__ == "__main__":
    print("📡 Fetching recent data for BTC/USDC...")
    since = int((datetime.utcnow() - timedelta(days=3)).timestamp() * 1000)
    df = fetch_ohlcv_binance("BTC/USDC", "1h", since)

    if df.empty:
        print("❌ No data available.")
    else:
        prediction, confidence = predict_latest(df)
        print(f"🤖 Hull Prediction: {prediction} | Confidence: {confidence:.2f}")

        # Optional: Save prediction
        output_row = {
            "timestamp": df["datetime"].iloc[-1],
            "prediction": prediction,
            "confidence": confidence
        }
        os.makedirs("predictions", exist_ok=True)
        pd.DataFrame([output_row]).to_csv(
            "predictions/hull_predictions.csv",
            mode="a",
            header=not os.path.exists("predictions/hull_predictions.csv"),
            index=False
        )
        print("📄 Saved prediction to predictions/hull_predictions.csv")
