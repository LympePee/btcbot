# btcbot/training/filtering/filter_macd_bb_signals.py

import pandas as pd
import os

# === Paths ===
INPUT_CSV = "data/training_sets/macd_bb_signals.csv"  # Unfiltered signals
OUTPUT_CSV = "data/training_sets/macd_bb_filtered_signals.csv"

# === Load Data ===
df = pd.read_csv(INPUT_CSV)

# === Compute return
df["return"] = (df["price_after_4h"] - df["entry_price"]) / df["entry_price"]

# === Filtering logic ===
filtered_df = df[
    (df["bb_breakout_pct"].abs() >= 0.012) &         # Clear BB breakout
    (df["macd_histogram"].abs() > 0.0008) &          # MACD momentum not flat
    (df["rsi"] >= 72) &                              # Strong RSI confirmation
    (df["return"].abs() >= 0.01)                     # At least 1% move (signal validation)
]

# Optional: drop return if not needed later
filtered_df.drop(columns=["return"], inplace=True)

# === Save ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
filtered_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Filtered MACD+BB signals saved to: {OUTPUT_CSV}")
print(f"📊 Retained signals: {len(filtered_df)} out of {len(df)} total")
