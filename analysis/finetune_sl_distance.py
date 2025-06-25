# analysis/finetune_sl_distance.py

import os
import pandas as pd
import numpy as np

IN_PATH = "data/training_sets/stoploss_labeled_signals.csv"
OUT_PATH = "analysis/stoploss_sl_distance_report.csv"

df = pd.read_csv(IN_PATH)

# Υπολογισμός ποσοστιαίας απόστασης SL από entry
df["sl_distance_pct"] = (df["sl_price"] - df["entry_price"]) / df["entry_price"] * 100

# Ομαδοποίηση σε bins κάθε 0.1%
bins = np.arange(df["sl_distance_pct"].min(), df["sl_distance_pct"].max(), 0.1)
df["sl_bin"] = pd.cut(df["sl_distance_pct"], bins)

# Υπολογισμός success rate ανά bin
grouped = df.groupby("sl_bin")["outcome"].agg(["count", "mean"]).reset_index()
grouped.rename(columns={"mean": "success_rate"}, inplace=True)

# Αποθήκευση αποτελεσμάτων
os.makedirs("analysis", exist_ok=True)
grouped.to_csv(OUT_PATH, index=False)
print(f"✅ Analysis saved to {OUT_PATH}")

# Preview: Εμφάνιση 10 πρώτων γραμμών με τα bins και το success rate
print("\nSample:")
print(grouped.head(10))
