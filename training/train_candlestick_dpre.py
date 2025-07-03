import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

label_map = {"bullish": 0, "bearish": 1, "sideways": 2}

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

class MultiTFPatternCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
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

def train_model(train_path, val_path, model_path, batch_size, epochs, lr):
    print("📂 Loading data...")
    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    train_ds = CandlestickDataset(train_data)
    val_ds = CandlestickDataset(val_data)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("❌ No valid training or validation data found.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTFPatternCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"🏋️ Training model for {epochs} epochs (batch={batch_size}, lr={lr})...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (x1h, x4h, x1d), y in train_loader:
            x1h, x4h, x1d, y = x1h.to(device), x4h.to(device), x1d.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x1h, x4h, x1d)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    print("🧪 Evaluating on validation set...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for (x1h, x4h, x1d), y in val_loader:
            x1h, x4h, x1d = x1h.to(device), x4h.to(device), x1d.to(device)
            out = model(x1h, x4h, x1d)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(y.numpy())

    print("📊 Classification Report (val):")
    print(classification_report(targets, preds, target_names=["bullish", "bearish", "sideways"]))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, required=True, help="Path to training set (.jsonl)")
    parser.add_argument("--val_set", type=str, required=True, help="Path to validation set (.jsonl)")
    parser.add_argument("--model_path", type=str, default="models/candlestick_dpre_model.pt", help="Output model path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    train_model(args.train_set, args.val_set, args.model_path, args.batch_size, args.epochs, args.lr)
