#feedback_logger.py
import os
import json
from datetime import datetime

FEEDBACK_PATH = "data/training_sets/feedback_patterns_candlestick.jsonl"

def log_feedback(
    candle_sequences: dict,
    predicted: str,
    confidence: float,
    true: str,
    timestamp: str = None,
):
    entry = {
        "timestamp": timestamp or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "1h_sequence": candle_sequences["1h_sequence"],
        "4h_sequence": candle_sequences["4h_sequence"],
        "1d_sequence": candle_sequences["1d_sequence"],
        "predicted": predicted,
        "true": true,
        "confidence": confidence,
    }
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"✅ Feedback logged at {entry['timestamp']}")

# ====== ΠΑΡΑΔΕΙΓΜΑ ======
if __name__ == "__main__":
    # Τρέχεις αυτό το log 4 ώρες ΜΕΤΑ το inference!
    fake_sequences = {
        "1h_sequence": [[0, 1, 2, 3, 4]] * 20,
        "4h_sequence": [[0, 1, 2, 3, 4]] * 20,
        "1d_sequence": [[0, 1, 2, 3, 4]] * 20,
    }
    log_feedback(
        fake_sequences,
        predicted="bullish",
        confidence=0.73,
        true="bearish",
        timestamp="2025-06-28T16:00:00Z"
    )
