#training/evaluate_candlestick_dpre.py
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np

label_map = {"bullish": 0, "bearish": 1, "sideways": 2}
inv_label_map = {v: k for k, v in label_map.items()}

class CandlestickDataset(Dataset):
    def __init__(self, data):
        self.x1h = []
        self.x4h = []
        self.x1d = []
        self.labels = []
        for item in data:
            if "target" not in item or item["target"] not in label_map:
                continue
            self.x1h.append(item["1h_sequence"])
            self.x4h.append(item["4h_sequence"])
            self.x1d.append(item["1d_sequence"])
            self.labels.append(label_map[item["target"]])
        self.x1h = torch.tensor(self.x1h, dtype=torch.float32)
        self.x4h = torch.tensor(self.x4h, dtype=torch.float32)
        self.x1d = torch.tensor(self.x1d, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.x1h[idx], self.x4h[idx], self.x1d[idx]), self.labels[idx]

class MultiTFPatternCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(5, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 3)
        )

    def forward(self, x1h, x4h, x1d):
        x1h = self.cnn(x1h.permute(0, 2, 1)).squeeze(-1)
        x4h = self.cnn(x4h.permute(0, 2, 1)).squeeze(-1)
        x1d = self.cnn(x1d.permute(0, 2, 1)).squeeze(-1)
        x = torch.cat([x1h, x4h, x1d], dim=1)
        return self.fc(x)

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def evaluate_model(eval_path, model_path, batch_size=64):
    data = load_jsonl(eval_path)
    ds = CandlestickDataset(data)
    loader = DataLoader(ds, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTFPatternCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for (x1h, x4h, x1d), y in loader:
            x1h, x4h, x1d = x1h.to(device), x4h.to(device), x1d.to(device)
            out = model(x1h, x4h, x1d)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(y.numpy())
    print("📊 Classification Report (eval):")
    print(classification_report(targets, preds, target_names=["bullish", "bearish", "sideways"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    evaluate_model(args.eval_set, args.model_path, args.batch_size)
