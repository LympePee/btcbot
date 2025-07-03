# training/retrain_candlestick_dpre.py
import os
import json
import argparse
import shutil
import torch
from train_candlestick_dpre import train_model, load_jsonl, label_map
from sklearn.metrics import classification_report, f1_score

# === FILE PATHS ===
CLEAN_PATH = "data/training_sets/candlestick_dpre_training_set.jsonl"
FEEDBACK_PATH = "data/training_sets/candlestick_dpre_feedback.jsonl"
MERGED_PATH = "data/training_sets/candlestick_dpre_merged.jsonl"
MODEL_CLEAN = "models/candlestick_dpre_model_clean.pt"
MODEL_FB = "models/candlestick_dpre_model_feedback.pt"
BEST_MODEL = "models/candlestick_dpre_model.pt"
MIN_DELTA = 0.002  # Require at least 0.2% F1 improvement

def merge_datasets(clean_path, feedback_path, out_path):
    print(f"🪄 Merging training & feedback data...")
    data = load_jsonl(clean_path)
    if os.path.exists(feedback_path):
        feedback = load_jsonl(feedback_path)
        # Remove feedback duplicates (by '1h_sequence' as proxy)
        seen = set(tuple(json.dumps(x["1h_sequence"])) for x in data)
        new_fb = [x for x in feedback if tuple(json.dumps(x["1h_sequence"])) not in seen]
        merged = data + new_fb
        print(f"   ➜ {len(data)} original, {len(new_fb)} feedback, {len(merged)} total samples.")
    else:
        merged = data
        print(f"   ➜ {len(data)} samples, no feedback found.")
    with open(out_path, "w") as f:
        for item in merged:
            f.write(json.dumps(item) + "\n")
    return out_path

def eval_model(model_path, test_data_path):
    from train_candlestick_dpre import CandlestickDataset, MultiTFPatternCNN
    dataset = CandlestickDataset(load_jsonl(test_data_path))
    if len(dataset) < 10:
        raise ValueError("Not enough test samples for eval!")
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTFPatternCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (x1h, x4h, x1d), y in loader:
            x1h, x4h, x1d = x1h.to(device), x4h.to(device), x1d.to(device)
            out = model(x1h, x4h, x1d)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(classification_report(all_labels, all_preds, target_names=["bullish", "bearish", "sideways"]))
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", type=str, default="data/training_sets/candlestick_dpre_evalset.jsonl",
                        help="Validation/test set for performance check")
    parser.add_argument("--prod_model", type=str, default=BEST_MODEL, help="Path for production model")
    args = parser.parse_args()

    # 1. Clean retrain
    print("\n--- Retrain (CLEAN DATA ONLY) ---")
    train_model(data_path=CLEAN_PATH, model_path=MODEL_CLEAN)
    f1_clean = eval_model(MODEL_CLEAN, args.eval_set)

    # 2. Feedback retrain
    print("\n--- Retrain (CLEAN + FEEDBACK) ---")
    merge_datasets(CLEAN_PATH, FEEDBACK_PATH, MERGED_PATH)
    train_model(data_path=MERGED_PATH, model_path=MODEL_FB)
    f1_fb = eval_model(MODEL_FB, args.eval_set)

    print(f"\n📝 F1 macro (clean): {f1_clean:.4f} | F1 macro (feedback): {f1_fb:.4f}")
    if f1_fb > f1_clean + MIN_DELTA:
        shutil.copy(MODEL_FB, args.prod_model)
        print(f"🚀 Feedback model promoted to production!")
    else:
        shutil.copy(MODEL_CLEAN, args.prod_model)
        print(f"⚠️ Clean model kept in production (feedback not better).")

if __name__ == "__main__":
    main()
