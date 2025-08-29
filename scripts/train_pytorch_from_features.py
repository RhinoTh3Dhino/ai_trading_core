# -*- coding: utf-8 -*-
"""
Træn en simpel PyTorch-model på dine eksisterende features-CSV og gem:
- models/best_pytorch_model.pt
- models/best_pytorch_features.json

Matcher arkitekturen som bot.engine.py forventer.
Kører som standard på 'auto' features (finder {SYMBOL}_{TF}_latest.csv).
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- PROJECT_ROOT (fallback) ---
try:
    from utils.project_path import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = Path(PROJECT_ROOT) / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- find auto features (samme logik som i engine fallback) ----
def ensure_latest(symbol: str = "BTCUSDT", timeframe: str = "1h"):
    outdir = Path(PROJECT_ROOT) / "outputs" / "feature_data"
    outdir.mkdir(parents=True, exist_ok=True)
    candidate = outdir / f"{symbol}_{timeframe}_latest.csv"
    if candidate.exists():
        return candidate
    alts = sorted(outdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if alts:
        return alts[0]
    raise FileNotFoundError("Ingen features-CSV fundet i outputs/feature_data/")

def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def load_dataframe(features_path: str) -> pd.DataFrame:
    with open(features_path, "r", encoding="utf-8") as f:
        first = f.readline()
    if str(first).startswith("#"):
        df = pd.read_csv(features_path, skiprows=1)
    else:
        df = pd.read_csv(features_path)
    # timestamp normalisering
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    if "timestamp" in df.columns:
        df["timestamp"] = _ensure_datetime(df["timestamp"])
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def pick_feature_columns(df: pd.DataFrame):
    """
    Brug samme filosofi som engine: tag alle numeriske kolonner
    bortset fra 'target' og helt tydelige meta-kolonner.
    """
    drop_cols = {"timestamp", "datetime", "target", "signal", "signal_ml", "signal_dl", "signal_ensemble"}
    num_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    # hvis "regime" findes men ikke er numerisk: map den til -1/0/1 (som engine)
    if "regime" in df.columns and not np.issubdtype(df["regime"].dtype, np.number):
        regime_map = {"bull": 1, "neutral": 0, "bear": -1}
        df["regime"] = df["regime"].map(regime_map).fillna(0)
        if "regime" not in num_cols:
            num_cols.append("regime")
    return num_cols

class TradingNet(nn.Module):
    # Matcher din engine.py (2 skjulte lag á 64, ReLU, output_dim=2)
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

def chronological_split(X, y, val_ratio=0.2):
    n = len(X)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])

def train(
    df: pd.DataFrame,
    feature_cols,
    out_model_path: Path,
    out_features_path: Path,
    device: str = "cpu",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    # klargør X/y (ingen skalering – matcher engine inference)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    if "target" in df.columns:
        y = pd.to_numeric(df["target"], errors="coerce").fillna(0).values.astype(np.int64)
        # sikkerhed: clamp til {0,1}
        y = np.clip(y, 0, 1)
    else:
        raise ValueError("Din CSV mangler 'target' kolonnen. Tilføj den eller lav labels først.")

    # kronologisk split (undgå lækage i tidsserier)
    (X_tr, y_tr), (X_va, y_va) = chronological_split(X, y, val_ratio=0.2)

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TradingNet(input_dim=X.shape[1]).to(device)
    # klassevægt hvis skæv fordeling
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    if pos == 0 or neg == 0:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    else:
        # vægt negativ/positiv invers proportional med frekvens
        w0 = pos / (pos + neg)
        w1 = neg / (pos + neg)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_va_loss = float("inf")
    best_state = None
    patience = 6
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * len(xb)
        tr_loss /= max(1, len(ds_tr))

        # val
        model.eval()
        va_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += float(loss.item()) * len(xb)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                total += int(len(yb))
        va_loss /= max(1, len(ds_va))
        va_acc = correct / max(1, total)

        print(f"[{epoch:03d}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.3f}")

        if va_loss < best_va_loss - 1e-6:
            best_va_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] Ingen forbedring i {patience} epoker. Stopper.")
                break

    if best_state is None:
        best_state = model.state_dict()

    # gem model (state_dict) og features-listen (json)
    torch.save(best_state, out_model_path)
    with open(out_features_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"✅ Gemte model til: {out_model_path}")
    print(f"✅ Gemte feature-liste til: {out_features_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="auto", help="Sti til features CSV eller 'auto'")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--interval", type=str, default="1h")
    ap.add_argument("--device", type=str, default=None, help="'cuda' eller 'cpu' (auto hvis None)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # find features-path
    if args.features.lower() == "auto" or not os.path.exists(args.features):
        feat_path = ensure_latest(args.symbol, args.interval)
        print(f"🧩 AUTO features → {feat_path}")
    else:
        feat_path = Path(args.features)
        print(f"📄 Bruger features → {feat_path}")

    df = load_dataframe(str(feat_path))
    if "target" not in df.columns:
        raise SystemExit("❌ CSV mangler 'target'. Tilføj target-kolonnen før træning.")

    feature_cols = pick_feature_columns(df)
    print(f"[INFO] Features ({len(feature_cols)}): {feature_cols}")

    out_model = MODELS_DIR / "best_pytorch_model.pt"
    out_feats = MODELS_DIR / "best_pytorch_features.json"

    train(
        df,
        feature_cols,
        out_model_path=out_model,
        out_features_path=out_feats,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

if __name__ == "__main__":
    main()
