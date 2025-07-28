"""
trainers/train_lightgbm.py

Træner en LightGBM-model til trading-signaler (klassifikation)
- Simpel kernepipeline til hurtige baseline-tests
- Automatisk feature selection, gemmer model og accuracy
"""

# Sikrer korrekt sys.path – projektroden lægges ind først
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent  # AUTO-FIXED PATHLIB
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.project_path import ensure_project_root
ensure_project_root()

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from utils.telegram_utils import send_message

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_lightgbm_model.txt")

def load_csv_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    skiprows = 1 if str(first_line).startswith("#") else 0
    return pd.read_csv(file_path, skiprows=skiprows)

def train_lightgbm_model(data_path, target_col="target", n_estimators=100, learning_rate=0.1, save_model=True):
    print(f"[INFO] Læser data fra: {data_path}")
    df = load_csv_auto(data_path)

    if target_col not in df.columns:
        print(f"[ADVARSEL] Target-kolonne '{target_col}' ikke fundet – dummy target oprettes.")
        df[target_col] = 0  # Dummy target, kun til pipeline-test

    X = df.drop(columns=[target_col, "timestamp"], errors="ignore").select_dtypes(include=[np.number])
    y = df[target_col]

    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Træner model på {len(X_train)} samples, validerer på {len(X_val)} samples")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"✅ Val accuracy: {acc:.3f}")

    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.booster_.save_model(MODEL_PATH)
        print(f"[INFO] Model gemt: {MODEL_PATH}")

    try:
        send_message(f"✅ LGBM training færdig | Val acc: {acc:.3f}")
    except Exception as e:
        print(f"[ADVARSEL] Telegram fejl: {e}")

    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Træn LightGBM baseline-model til trading-signaler")
    parser.add_argument("--data", type=str, required=True, help="Path til features-data (.csv)")
    parser.add_argument("--target", type=str, default="target", help="Target-kolonne")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    train_lightgbm_model(
        data_path=args.data,
        target_col=args.target,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        save_model=not args.no_save
    )
