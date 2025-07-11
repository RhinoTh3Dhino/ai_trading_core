import sys
import os
import argparse
import json
import pandas as pd
import numpy as np

# === Central logging ===
from utils.log_utils import log_device_status
from utils.telegram_utils import send_message

# === CLI-argumenter: gør engine.py CLI-klar ===
parser = argparse.ArgumentParser(description="AI Trading Engine")
parser.add_argument("--features", type=str, required=True, help="Sti til feature-fil (CSV)")
parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
parser.add_argument("--interval", type=str, default="1h", help="Tidsinterval (fx 1h, 4h)")
parser.add_argument("--model_type", type=str, default="ml", choices=["ml", "dl", "ensemble"], help="Vælg model-type")
parser.add_argument("--device", type=str, default=None, help="PyTorch device ('cuda'/'cpu'), auto hvis None")
args = parser.parse_args()

SYMBOL = args.symbol
INTERVAL = args.interval

# === Device-detection og logging (centralt) ===
device_info = log_device_status(
    context="engine",
    extra={
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "model_type": args.model_type
    },
    telegram_func=send_message,
    print_console=True
)
# Typisk får du "device_str": "cuda" eller "cpu" retur
DEVICE_STR = device_info.get("device_str", "cpu")

# === Importér alt fra din tidligere engine (ingen ændring på forretningslogik) ===
from bot.monitor import ResourceMonitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from versions import (
        PIPELINE_VERSION, PIPELINE_COMMIT,
        FEATURE_VERSION, ENGINE_VERSION, ENGINE_COMMIT, MODEL_VERSION, LABEL_STRATEGY
    )
except ImportError:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = ENGINE_COMMIT = MODEL_VERSION = LABEL_STRATEGY = "unknown"

from backtest.backtest import run_backtest, calc_backtest_metrics
from backtest.metrics import evaluate_strategies
from visualization.plot_backtest import plot_backtest
from visualization.plot_drawdown import plot_drawdown
from visualization.plot_strategy_score import plot_strategy_scores
from utils.telegram_utils import (
    send_image, send_message, send_performance_report
)
from utils.robust_utils import safe_run

from ensemble.majority_vote_ensemble import majority_vote_ensemble
from ensemble.weighted_vote_ensemble import weighted_vote_ensemble

from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals
from strategies.ema_cross_strategy import ema_cross_signals

from visualization.viz_feature_importance import plot_feature_importance
from utils.feature_logging import log_top_features_to_md, log_top_features_csv, send_top_features_telegram

import torch

# === PyTorch-model & paths ===
MODEL_DIR = "models"
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "best_pytorch_model.pt")

class TradingNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

def load_pytorch_model(feature_dim, model_path=PYTORCH_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"❌ PyTorch-model ikke fundet: {model_path}")
        return None
    model = TradingNet(input_dim=feature_dim, output_dim=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE_STR))
    model.eval()
    model.to(DEVICE_STR)
    print(f"✅ PyTorch-model indlæst fra {model_path} på {DEVICE_STR}")
    return model

def pytorch_predict(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(DEVICE_STR)
        logits = model(X_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

# === Hjælpefunktioner (som tidligere) ===

GRAPH_DIR = "graphs/"
DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 0.7, 0.4, 1.0]
RETRAIN_WINRATE_THRESHOLD = 0.30
RETRAIN_PROFIT_THRESHOLD = 0.0
MAX_RETRAINS = 3

USE_REGIME_FILTER = False   # True = produktion, False = debug/test
ADAPTIVE_WINRATE_THRESHOLD = 0.0

def read_features_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    return df

# ... (resten af hjælpefunktionerne er som tidligere - behold dem fra din kode!)

def main(threshold=DEFAULT_THRESHOLD, weights=DEFAULT_WEIGHTS, FORCE_DEBUG=False):
    monitor = ResourceMonitor(
        ram_max=85, cpu_max=90, gpu_max=95, gpu_temp_max=80, check_interval=10,
        action="pause", log_file="outputs/debug/resource_log.csv"
    )
    monitor.start()
    retrain_count = 0
    seed = None

    try:
        DATA_PATH = args.features
        print("🔄 Indlæser features:", DATA_PATH)
        df = read_features_auto(DATA_PATH)
        print(f"✅ Data indlæst ({len(df)} rækker)")
        print("Kolonner:", list(df.columns))

        # === ML, DL, ensemble flow baseret på model_type ===
        feature_cols = [col for col in df.columns if col not in ("timestamp", "target", "regime", "signal")]

        ml_signals = None
        probas = None

        if args.model_type == "dl":
            print("🔄 Loader PyTorch DL-model ...")
            model = load_pytorch_model(feature_dim=len(feature_cols))
            if model is not None:
                ml_signals, probas = pytorch_predict(model, df[feature_cols])
                if probas.shape[1] == 2:
                    signal_proba = probas[:, 1]    # probability for class 1 (BUY)
                    ml_signals = (signal_proba > DEFAULT_THRESHOLD).astype(int)
                print("✅ PyTorch DL-inference klar!")
            else:
                print("❌ Ingen DL-model fundet – fallback til random signaler")
                ml_signals = np.random.choice([0, 1], size=len(df))
        elif args.model_type == "ml":
            print("🛠️ ML-inference ikke implementeret i dette eksempel – fallback til random")
            ml_signals = np.random.choice([0, 1], size=len(df))
        elif args.model_type == "ensemble":
            print("🔄 Loader PyTorch DL-model (til ensemble)...")
            model = load_pytorch_model(feature_dim=len(feature_cols))
            if model is not None:
                dl_signals, probas = pytorch_predict(model, df[feature_cols])
                dl_signals = (probas[:, 1] > DEFAULT_THRESHOLD).astype(int)
            else:
                dl_signals = np.random.choice([0, 1], size=len(df))
            ml_signals = np.random.choice([0, 1], size=len(df))
            rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
            macd_signals = macd_cross_signals(df)
            ema_signals = ema_cross_signals(df)
            ensemble_signals = majority_vote_ensemble(dl_signals, ml_signals, rsi_signals, macd_signals, ema_signals)
            df["signal"] = ensemble_signals
        else:
            print("Ukendt model_type – stopper.")
            return

        # === Resten af pipeline: signaler, backtest, metrics, plots, telegram osv. ===
        # (Kopier blot din eksisterende kode ind her fra main())
        # Bemærk: Hvis du bruger ensemble, skal df["signal"] sættes korrekt

        # ... Din eksisterende pipeline ... (backtest, plot, telegram, retrain osv.)

        print("🎉 Pipeline afsluttet uden fejl!")

    finally:
        monitor.stop()

if __name__ == "__main__":
    safe_run(lambda: main(threshold=DEFAULT_THRESHOLD, weights=DEFAULT_WEIGHTS, FORCE_DEBUG=False))
