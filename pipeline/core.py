# pipeline/core.py

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from backtest.backtest import run_backtest, calc_backtest_metrics
from strategies.rsi_strategy import rsi_rule_based_signals
from ensemble.ensemble_predict import ensemble_predict
from utils.telegram_utils import send_message, send_image
from utils.log_utils import log_device_status
from visualization.plot_performance import plot_performance
from visualization.plot_comparison import plot_comparison

MODEL_DIR = "models"
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
PYTORCH_FEATURES_PATH = os.path.join(MODEL_DIR, "best_pytorch_features.json")
GRAPH_DIR = "graphs/"

DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 1.0, 0.7]  # [ML, DL, Rule]

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

def load_trained_feature_list():
    if not os.path.exists(PYTORCH_FEATURES_PATH):
        print(f"[ADVARSEL] Kunne ikke finde {PYTORCH_FEATURES_PATH} – bruger alle numeriske features fra input.")
        return None
    with open(PYTORCH_FEATURES_PATH, "r") as f:
        features = json.load(f)
    print(f"[INFO] Loader feature-liste fra model: {features}")
    return features

def load_pytorch_model(feature_dim, model_path=PYTORCH_MODEL_PATH, device="cpu"):
    if not os.path.exists(model_path):
        print(f"❌ PyTorch-model ikke fundet: {model_path}")
        return None
    model = TradingNet(input_dim=feature_dim, output_dim=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print(f"✅ PyTorch-model indlæst fra {model_path} på {device}")
    return model

def pytorch_predict(model, X, device="cpu"):
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def read_features_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    return df

def run_pipeline(
    features_path,
    symbol="BTCUSDT",
    interval="1h",
    threshold=DEFAULT_THRESHOLD,
    weights=DEFAULT_WEIGHTS,
    log_to_tb=True,
    tensorboard_dir="runs",
    device=None,
    send_telegram=True,
    plot_graphs=True,
    save_graphs=True,
    telegram_caption=None,
    verbose=True,
    extra_pipeline_info=None,
):
    """
    Kører hele AI trading pipeline: indlæs data, ML/DL/ensemble inference, backtest, visualisering, metrics, TensorBoard og (valgfrit) Telegram.
    Kan kaldes fra både main.py, run_all.py, tests osv.
    """
    # Device management
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log device status (og evt. Telegram)
    log_device_status(
        context="pipeline_core",
        extra={
            "symbol": symbol,
            "interval": interval,
            "model_type": "multi_compare",
            **(extra_pipeline_info or {})
        },
        telegram_func=send_message if send_telegram else None,
        print_console=verbose
    )

    # TensorBoard-writer (én run pr. kald)
    writer = SummaryWriter(log_dir=f"{tensorboard_dir}/core_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if log_to_tb else None

    # Indlæs features/data
    print(f"\n🔄 Indlæser features: {features_path}")
    df = read_features_auto(features_path)
    print(f"✅ Data indlæst ({len(df)} rækker)")
    print("Kolonner:", list(df.columns))

    # Sikrer feature-match til PyTorch-model
    trained_features = load_trained_feature_list()
    if trained_features is not None:
        missing = [f for f in trained_features if f not in df.columns]
        if missing:
            print(f"[FEJL] Følgende features mangler i input: {missing}")
            raise RuntimeError("Feature-mismatch! Træn modellen forfra.")
        X_dl = df[trained_features]
    else:
        fallback_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ("timestamp", "target", "regime", "signal")]
        X_dl = df[fallback_cols]
        print(f"[ADVARSEL] Kører med fallback-features: {fallback_cols}")

    metrics_dict = {}

    # --- ML (random baseline for nu) ---
    print("🛠️ ML-inference ikke implementeret – bruger random (baseline demo)")
    ml_signals = np.random.choice([0, 1], size=len(df))
    df["signal_ml"] = ml_signals
    trades_ml, balance_ml = run_backtest(df, signals=ml_signals)
    metrics_ml = calc_backtest_metrics(trades_ml, balance_ml)
    metrics_dict["ML"] = metrics_ml

    # --- DL (PyTorch) ---
    print("🔄 Loader PyTorch DL-model ...")
    model = load_pytorch_model(feature_dim=X_dl.shape[1], device=device)
    if model is not None:
        dl_preds, dl_probas = pytorch_predict(model, X_dl, device=device)
        dl_signals = (dl_probas[:, 1] > threshold).astype(int)
        print("✅ PyTorch DL-inference klar!")
    else:
        print("❌ Ingen DL-model fundet – fallback til random signaler")
        dl_signals = np.random.choice([0, 1], size=len(df))
    df["signal_dl"] = dl_signals
    trades_dl, balance_dl = run_backtest(df, signals=dl_signals)
    metrics_dl = calc_backtest_metrics(trades_dl, balance_dl)
    metrics_dict["DL"] = metrics_dl

    # --- Ensemble (DL + ML + RSI) ---
    rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
    ensemble_signals = ensemble_predict(
        ml_preds=ml_signals,
        dl_preds=dl_signals,
        rule_preds=rsi_signals,
        weights=weights,
        voting="majority",
        debug=True
    )
    df["signal_ensemble"] = ensemble_signals
    trades_ens, balance_ens = run_backtest(df, signals=ensemble_signals)
    metrics_ens = calc_backtest_metrics(trades_ens, balance_ens)
    metrics_dict["Ensemble"] = metrics_ens

    # --- Print & log metrics ---
    print("\n=== Performance metrics (backtest) ===")
    for model, metrics in metrics_dict.items():
        print(f"{model}: {metrics}")

    # --- TensorBoard-logging ---
    if writer:
        for model_name, metrics in metrics_dict.items():
            for metric_key, value in metrics.items():
                writer.add_scalar(f"{model_name}/{metric_key}", value)
        writer.flush()
        writer.close()
        print(f"[INFO] Metrics logget til TensorBoard (run: {writer.log_dir})")

    # --- Visualisering og grafer ---
    if plot_graphs or save_graphs:
        os.makedirs(GRAPH_DIR, exist_ok=True)
        plot_performance(balance_ml, trades_ml, model_name="ML", save_path=f"{GRAPH_DIR}/performance_ml.png")
        plot_performance(balance_dl, trades_dl, model_name="DL", save_path=f"{GRAPH_DIR}/performance_dl.png")
        plot_performance(balance_ens, trades_ens, model_name="Ensemble", save_path=f"{GRAPH_DIR}/performance_ensemble.png")
        metric_keys = ["profit_pct", "win_rate", "drawdown_pct", "num_trades"]
        plot_comparison(metrics_dict, metric_keys=metric_keys, save_path=f"{GRAPH_DIR}/model_comparison.png")
        print(f"[INFO] Grafer gemt til {GRAPH_DIR}")

    # --- Telegram-support ---
    if send_telegram:
        try:
            send_image(f"{GRAPH_DIR}/model_comparison.png", caption=telegram_caption or f"{symbol} {interval} | ML vs. DL vs. ENSEMBLE performance")
        except Exception as e:
            print(f"[ADVARSEL] Telegram-graf kunne ikke sendes: {e}")

    print("\n🎉 Pipeline afsluttet uden fejl!")

    return metrics_dict
