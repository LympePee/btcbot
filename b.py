from collections import Counter
import json
with open("data/training_sets/candlestick_dpre_training_set.jsonl") as f:
    labels = [json.loads(line)["target"] for line in f]
print(Counter(labels))
