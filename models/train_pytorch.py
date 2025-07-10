"""
models/train_pytorch.py

Træner en PyTorch neural net model til trading-signaler (klassifikation)
- Bruges direkte fra CLI eller fra engine.py/andre scripts.
- Gemmer og loader model (.pt), log og metrics.
- Automatisk brug af GPU hvis muligt.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.telegram_utils import send_message

# === Konfiguration ===
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
LOG_PATH = os.path.join(MODEL_DIR, "train_log_pytorch.txt")

def log_to_file(line, prefix="[INFO] "):
    os.makedirs("logs", exist_ok=True)
    with open("logs/bot.log", "a", encoding="utf-8") as logf:
        logf.write(prefix + line)

def log_device_status(data_path, batch_size, epochs, lr):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch_version = torch.__version__
    python_version = platform.python_version()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_mem_alloc = torch.cuda.memory_allocated() // (1024**2)
        cuda_mem_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        status_line = (f"{now} | PyTorch {torch_version} | Python {python_version} | "
                       f"Device: GPU ({device_name}) | CUDA alloc: {cuda_mem_alloc} MB / {cuda_mem_total} MB | "
                       f"Data: {data_path} | Batch: {batch_size} | Epochs: {epochs} | LR: {lr}\n")
    else:
        status_line = (f"{now} | PyTorch {torch_version} | Python {python_version} | "
                       f"Device: CPU | Data: {data_path} | Batch: {batch_size} | Epochs: {epochs} | LR: {lr}\n")
    print(f"[BotStatus.md] {status_line.strip()}")
    with open("BotStatus.md", "a", encoding="utf-8") as f:
        f.write(status_line)
    log_to_file(status_line)
    try:
        send_message("🤖 " + status_line.strip())
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke sende til Telegram: {e}")

# === Automatisk CSV-loader (springer meta-header over) ===
def load_csv_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("[INFO] Meta-header fundet i CSV – loader med skiprows=1")
        return pd.read_csv(file_path, skiprows=1)
    else:
        return pd.read_csv(file_path)

# === PyTorch dataset ===
class TradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)  # long for classification

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Simpel MLP-model ===
class TradingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# === Træningsfunktion ===
def train_pytorch_model(
    data_path,
    target_col="target",
    batch_size=32,
    epochs=30,
    learning_rate=1e-3,
    test_size=0.2,
    random_state=42
):
    # === Device-logning (pro) ===
    log_device_status(data_path, batch_size, epochs, learning_rate)

    # === Data ===
    print(f"[INFO] Indlæser data fra: {data_path}")
    df = load_csv_auto(data_path)
    assert target_col in df.columns, f"target_col '{target_col}' ikke fundet!"

    # Fjern kolonner der ikke skal med
    drop_cols = [target_col, "timestamp"]
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    y = df[target_col]

    # === Kun numeriske features! ===
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        ignored = set(X.columns) - set(X_numeric.columns)
        print(f"[ADVARSEL] Ignorerer ikke-numeriske features: {ignored}")
    X = X_numeric

    print(f"[INFO] Features brugt til træning: {list(X.columns)}")
    print(f"[INFO] Unikke targets: {sorted(y.unique())}")
    print(f"[INFO] Target distribution: \n{y.value_counts()}")

    # --- Class weights for imbalanced data ---
    unique_classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"[INFO] Class weights (imbalance compensation): {weights}")

    # === Time-based split for tidsseriedata ===
    df = df.reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print("[INFO] Train slutter:", df.iloc[split_idx-1]["timestamp"] if "timestamp" in df.columns else split_idx-1)
    print("[INFO] Val starter:", df.iloc[split_idx]["timestamp"] if "timestamp" in df.columns else split_idx)
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")

    # --- Fordeling for debugging ---
    print("Target-fordeling (train):\n", y_train.value_counts(normalize=True))
    print("Target-fordeling (val):\n", y_val.value_counts(normalize=True))

    train_ds = TradingDataset(X_train, y_train)
    val_ds = TradingDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # === Model + optimering ===
    model = TradingNet(input_dim=X.shape[1], output_dim=len(unique_classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === Evaluer på val-set ===
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.numpy())
        acc = np.mean(np.array(y_pred) == np.array(y_true))

        print(f"[{epoch:02d}/{epochs}] Train loss: {train_loss:.4f} | Val acc: {acc:.3f}")

        # === Gem bedste model ===
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Ny bedste model gemt: {MODEL_PATH} (val_acc={acc:.3f})")

    # === Endelig rapport/log ===
    print("\n[INFO] Evaluering på valideringsdata:")
    print(classification_report(y_true, y_pred, zero_division=0))
    conf = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", conf)
    log_str = f"\n== Træningslog {datetime.now()} ==\n" \
              f"Model: {MODEL_PATH}\nVal_acc: {best_acc:.3f}\n" \
              f"Features: {list(X.columns)}\n\n" \
              f"Report:\n{classification_report(y_true, y_pred, zero_division=0)}\nConfusion:\n{conf}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_str)
    print(f"[INFO] Træningslog gemt til: {LOG_PATH}")

    # --- Ekstra debugging: print de første 20 prediction/target ---
    print("[DEBUG] Første 20 targets:", y_true[:20])
    print("[DEBUG] Første 20 preds:", y_pred[:20])

# === CLI-interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Træn PyTorch-model til trading (GPU/CPU)")
    parser.add_argument("--data", type=str, required=True, help="Sti til features-data (.csv)")
    parser.add_argument("--target", type=str, default="target", help="Navn på target-kolonne")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Antal epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split")
    args = parser.parse_args()

    train_pytorch_model(
        data_path=args.data,
        target_col=args.target,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        test_size=args.test_size,
    )
