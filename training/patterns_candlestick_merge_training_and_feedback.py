# patterns_candlestick_merge_training_and_feedback.py
# Author: Crypto Quant
# Συνδυάζει historical_patterns.jsonl + feedback_patterns_candlestick.jsonl
# Παράγει balanced/curriculum/hard-negative dataset για retrain Candlestick DPRE ML

import os
import json
import random
from collections import Counter

# === PATHS ===
HISTORICAL = "data/training_sets/historical_patterns.jsonl"
FEEDBACK = "data/training_sets/feedback_patterns_candlestick.jsonl"
OUTPUT = "data/training_sets/final_training_set_candlestick.jsonl"

CURRICULUM_EASY_ONLY = True   # Αν True, στο πρώτο retrain προτιμά εύκολα καθαρά signals

# --- Helper: φόρτωσε JSONL ως λίστα dicts ---
def load_jsonl(path):
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path} (skipping)")
        return []
    with open(path) as f:
        return [json.loads(x) for x in f]

def write_jsonl(path, data):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

# --- 1. Load data ---
hist = load_jsonl(HISTORICAL)
fb = load_jsonl(FEEDBACK)

print(f"Loaded {len(hist)} historical, {len(fb)} feedback samples.")

# --- 2. Curriculum: αρχικά πιο καθαρά σήματα (εκτός ambiguous) ---
def filter_easy_signals(samples):
    return [
        s for s in samples
        if (s.get("target") in ("bullish", "bearish", "sideways"))
    ]

# --- 3. Hard Negative Mining ---
def extract_hard_negatives(fb_samples):
    # Παίρνεις όσα true != predicted (δηλ. όπου μπέρδεψε το μοντέλο)
    return [s for s in fb_samples if "true" in s and "predicted" in s and s["true"] != s["predicted"]]

# --- 4. Pattern Diversity ---
def pattern_diversity_sampling(samples, N=2000):
    # Ισορροπημένο sampling από κάθε class
    classes = ["bullish", "bearish", "sideways"]
    pools = {cls: [s for s in samples if s.get("target") == cls] for cls in classes}
    min_class = min(len(pools[cls]) for cls in classes)
    N = min(N, min_class)
    result = []
    for cls in classes:
        result.extend(random.sample(pools[cls], N))
    return result

# --- 5. Compose Final Training Set ---
final = []

# Curriculum: Ξεκινάμε με εύκολα & καθαρά
easy_hist = filter_easy_signals(hist)
easy_fb = filter_easy_signals(fb)

# Extra hard negatives (βάζουμε *2 βάρος)
hard_fb = extract_hard_negatives(fb)
final.extend(easy_hist)
final.extend(easy_fb)
final.extend(hard_fb)
final.extend(hard_fb)  # duplication for emphasis

print(f"Easy hist: {len(easy_hist)} | Easy fb: {len(easy_fb)} | Hard negatives: {len(hard_fb)}")

# --- 6. Pattern Diversity (προαιρετικά, αν έχεις μεγάλο dataset) ---
# final = pattern_diversity_sampling(final, N=2000)

random.shuffle(final)
print(f"Final training set size: {len(final)}")

# --- 7. Save Output ---
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
write_jsonl(OUTPUT, final)
print(f"✅ Saved merged dataset to {OUTPUT}")

# --- 8. Counter log για sanity check ---
print("Sample class distribution:", Counter([s.get("target", "X") for s in final]))
