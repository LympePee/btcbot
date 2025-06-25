import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

# Add project root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from btcbot.utils.fetch_binance_save_csv import fetch_binance_ohlcv
from btcbot.features.hull_features import calculate_hma
from btcbot.features.stoploss_features import extract_stoploss_features

DATA_DIR = "btcbot/data/historical"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
FUTURE_WINDOW = 72  # ώρες μετά το entry

def simulate_static_sl(entry_price, feature_row):
    ml_factor = compute_ml_factor_from_features(feature_row)
    sl = entry_price * (1 - (1.0 + ml_factor) / 100)
    return round(sl, 2), ml_factor

def compute_ml_factor_from_features(row):
    if row['rsi'] > 70:
        return +0.2
    elif row['rsi'] < 30:
        return -0.2
    return 0.0

def generate_hull_buy_signals(df: pd.DataFrame, hma_period: int = 21) -> pd.DataFrame:
    df = df.copy()
    df['hma'] = calculate_hma(df, hma_period)
    df['hma_shift'] = df['hma'].shift(1)
    df['signal'] = np.where(
        (df['hma'] > df['hma_shift']) & (df['hma_shift'] < df['hma_shift'].shift(1)),
        'buy',
        None
    )
    return df

def run_backtest_and_generate_training():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[INFO] Fetching historical data for {SYMBOL} ({TIMEFRAME})...")
    df = fetch_binance_ohlcv(SYMBOL, TIMEFRAME)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("[INFO] Applying Hull strategy to detect entries...")
    df_signals = generate_hull_buy_signals(df.copy())
    df_buys = df_signals[df_signals['signal'] == 'buy']

    training_data = []

    for idx, row in df_buys.iterrows():
        entry_time = row['timestamp']
        entry_price = row['close']

        future_data = df[
            (df['timestamp'] > entry_time) &
            (df['timestamp'] <= entry_time + timedelta(hours=FUTURE_WINDOW))
        ].copy()

        if future_data.empty or len(future_data) < 14:
            continue

        for i, current_row in future_data.iterrows():
            current_time = current_row['timestamp']
            snapshot = future_data.loc[:i].copy()

            if len(snapshot) < 14:
                continue  # Skip if not enough bars for indicators

            snapshot['entry_price'] = entry_price
            snapshot['entry_time'] = entry_time

            try:
                features_df = extract_stoploss_features(snapshot)
            except Exception as e:
                print(f"[WARN] Feature extraction failed: {e}")
                continue

            if features_df is None or not isinstance(features_df, pd.DataFrame) or features_df.empty:
                continue

            f_row = features_df.iloc[-1]
            sl_price, ml_factor = simulate_static_sl(entry_price, f_row)
            current_price = current_row['close']
            outcome = "hit" if current_price <= sl_price else "held"

            training_data.append({
                **f_row.to_dict(),
                "entry_price": entry_price,
                "sl_price": sl_price,
                "current_price": current_price,
                "ml_factor_target": ml_factor,
                "outcome": outcome
            })

    outpath = os.path.join(DATA_DIR, "sl_training_data.csv")
    pd.DataFrame(training_data).to_csv(outpath, index=False)
    print(f"[DONE] Training data saved to: {outpath}")

if __name__ == "__main__":
    run_backtest_and_generate_training()
