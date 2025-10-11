from pathlib import Path

from utils.project_path import PROJECT_ROOT

"""
models/optuna_tuner.py

Hyperparameter-tuning af PyTorch-model med Optuna (GPU/CPU)
- Genbruger train_pytorch.py funktioner.
- Logger alle trials til CSV, Telegram og konsol.
- Efter tuning gemmes bedste model automatisk.
"""

import argparse
import os
import sys
from datetime import datetime

# Sikrer, at models/ og projektrod er i path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).parent.parent  # AUTO-FIXED PATHLIB
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from trainers.train_pytorch import train_pytorch_model

# Telegram-integration (valgfrit)
try:
    from utils.telegram_utils import send_message
except ImportError:
    send_message = lambda msg: None

# AUTO PATH CONVERTED
OPTUNA_LOG_PATH = PROJECT_ROOT / "models" / "optuna_trials.csv"
# AUTO PATH CONVERTED
MODEL_PATH = PROJECT_ROOT / "models" / "best_pytorch_model.pt"


def optuna_objective(trial, data_path, target, test_size):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    epochs = trial.suggest_int("epochs", 10, 40)

    acc = train_pytorch_model(
        data_path=data_path,
        target_col=target,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        test_size=test_size,
        verbose=False,
        save_model=False,
    )
    # Log trial til CSV
    with open(OPTUNA_LOG_PATH, "a", encoding="utf-8") as f:
        line = f"{datetime.now()},{batch_size},{learning_rate:.5f},{hidden_dim},{n_layers},{dropout:.2f},{epochs},{acc:.4f}\n"
        f.write(line)
    return acc


def main():
    parser = argparse.ArgumentParser(description="Optuna-tuner til PyTorch trading-model")
    parser.add_argument("--data", type=str, required=True, help="Path til feature-data (CSV)")
    parser.add_argument("--target", type=str, default="target", help="Target-kolonne")
    parser.add_argument("--trials", type=int, default=20, help="Antal trials")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split")
    args = parser.parse_args()

    # Log header til CSV
    if not os.path.exists(OPTUNA_LOG_PATH):
        with open(OPTUNA_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(
                "datetime,batch_size,learning_rate,hidden_dim,n_layers,dropout,epochs,val_acc\n"
            )

    study = optuna.create_study(direction="maximize")
    print(f"🔍 Starter Optuna-tuning ({args.trials} trials) på {args.data}")
    send_message(f"🔍 Optuna tuning starter: {args.data} ({args.trials} trials)")

    def objective(trial):
        return optuna_objective(trial, args.data, args.target, args.test_size)

    study.optimize(objective, n_trials=args.trials)

    print("\n=== Optuna tuning færdig! ===")
    print("Bedste params:", study.best_trial.params)
    print("Bedste accuracy:", study.best_trial.value)
    send_message(
        f"🏆 Optuna tuning færdig! Bedste acc: {study.best_trial.value:.4f}\nParams: {study.best_trial.params}"
    )

    # Træn bedste model én gang med save_model=True
    best = study.best_trial.params
    final_acc = train_pytorch_model(
        data_path=args.data,
        target_col=args.target,
        batch_size=best["batch_size"],
        epochs=best["epochs"],
        learning_rate=best["lr"],
        hidden_dim=best["hidden_dim"],
        n_layers=best["n_layers"],
        dropout=best["dropout"],
        test_size=args.test_size,
        verbose=True,
        save_model=True,
    )
    print(f"✅ Bedste model trænet og gemt til: {MODEL_PATH} (val_acc={final_acc:.4f})")
    send_message(f"✅ Bedste model trænet og gemt til {MODEL_PATH}\nVal_acc={final_acc:.4f}")


if __name__ == "__main__":
    main()
