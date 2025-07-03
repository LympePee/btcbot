import json
import argparse
import os

def split_jsonl(input_path, train_path, val_path, eval_path, val_ratio=0.1, eval_ratio=0.05, shuffle=True):
    # Load data
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]

    if len(samples) < 20:
        raise ValueError("❌ Not enough samples to split!")

    if shuffle:
        from random import shuffle as rshuffle
        rshuffle(samples)

    total = len(samples)
    n_eval = int(total * eval_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_eval

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    eval_samples = samples[n_train + n_val:]

    # Save splits
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")
    with open(eval_path, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Split done! Training: {len(train_samples)}, Validation: {len(val_samples)}, Evaluation: {len(eval_samples)}")
    print(f"ℹ️  Files: {train_path} {val_path} {eval_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--train", type=str, required=True, help="Output train set path")
    parser.add_argument("--val", type=str, required=True, help="Output validation set path")
    parser.add_argument("--eval", type=str, required=True, help="Output evaluation set path")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--eval_ratio", type=float, default=0.05, help="Evaluation set ratio")
    parser.add_argument("--no_shuffle", action="store_true", help="Set to disable shuffle before split")
    args = parser.parse_args()

    split_jsonl(
        input_path=args.input,
        train_path=args.train,
        val_path=args.val,
        eval_path=args.eval,
        val_ratio=args.val_ratio,
        eval_ratio=args.eval_ratio,
        shuffle=not args.no_shuffle
    )
