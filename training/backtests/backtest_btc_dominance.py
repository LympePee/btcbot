import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from features.btc_dominance_features import get_btc_dominance_features

OUTPUT_CSV = "data/training_sets/btc_dominance_signals.csv"
START_DATE = "2017-01-01"
END_DATE = datetime.utcnow().strftime("%Y-%m-%d")
TIMEFRAME = "4h"

date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="4H")
signals = []

print(f"⏳ Generating BTC Dominance signals from {START_DATE} to {END_DATE}...")

for ts in date_range[:-2]:  # exclude last point (no future data for outcome)
    try:
        features = get_btc_dominance_features(ts)
        if features is None:
            continue

        future_ts = ts + timedelta(hours=4)
        future_features = get_btc_dominance_features(future_ts)
        if future_features is None:
            continue

        change = future_features["btc_dominance"] - features["btc_dominance"]
        outcome = int(change > 0.005)

        signal = {
            "timestamp": ts,
            "btc_dominance_change_4h": features["btc_dominance_change_4h"],
            "market_breadth_1h": features["market_breadth_1h"],
            "correlation_btc_vs_eth_bnb_xrp": features["correlation_btc_vs_eth_bnb_xrp"],
            "btc_price_change_4h": features["btc_price_change_4h"],
            "eth_price_change_4h": features["eth_price_change_4h"],
            "alt_avg_return_4h": features["alt_avg_return_4h"],
            "btc_volume_dominance": features["btc_volume_dominance"],
            "usdc.d_change": features["usdc.d_change"],
            "outcome": outcome
        }
        signals.append(signal)
    except Exception as e:
        print(f"⚠️ Error @ {ts}: {e}")
        continue

# === Save to CSV ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame(signals)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved BTC Dominance signals to: {OUTPUT_CSV}")
