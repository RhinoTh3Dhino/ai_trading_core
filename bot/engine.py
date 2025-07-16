# bot/engine.py

import sys
import os
import json
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.log_utils import log_device_status
from utils.telegram_utils import send_message, send_image, send_ensemble_metrics
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from bot.monitor import ResourceMonitor

try:
    from versions import (
        PIPELINE_VERSION, PIPELINE_COMMIT,
        FEATURE_VERSION, ENGINE_VERSION, ENGINE_COMMIT, MODEL_VERSION, LABEL_STRATEGY
    )
except ImportError:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = ENGINE_COMMIT = MODEL_VERSION = LABEL_STRATEGY = "unknown"

from backtest.backtest import run_backtest, calc_backtest_metrics
from utils.robust_utils import safe_run
from ensemble.ensemble_predict import ensemble_predict

from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals
from strategies.ema_cross_strategy import ema_cross_signals

import torch

MODEL_DIR = "models"
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
PYTORCH_FEATURES_PATH = os.path.join(MODEL_DIR, "best_pytorch_features.json")
LSTM_FEATURES_PATH = os.path.join(MODEL_DIR, "lstm_features.csv")
LSTM_SCALER_MEAN_PATH = os.path.join(MODEL_DIR, "lstm_scaler_mean.npy")
LSTM_SCALER_SCALE_PATH = os.path.join(MODEL_DIR, "lstm_scaler_scale.npy")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
ML_MODEL_PATH = os.path.join(MODEL_DIR, "best_ml_model.pkl")
ML_FEATURES_PATH = os.path.join(MODEL_DIR, "best_ml_features.json")

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
    if os.path.exists(PYTORCH_FEATURES_PATH):
        with open(PYTORCH_FEATURES_PATH, "r") as f:
            features = json.load(f)
        print(f"[INFO] Loader PyTorch feature-liste fra model: {features}")
        return features
    elif os.path.exists(LSTM_FEATURES_PATH):
        features = pd.read_csv(LSTM_FEATURES_PATH, header=None)[0].tolist()
        print(f"[INFO] Loader LSTM feature-liste fra model: {features}")
        return features
    else:
        print(f"[ADVARSEL] Ingen feature-liste fundet – bruger alle numeriske features fra input.")
        return None

def load_ml_model():
    if os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_FEATURES_PATH):
        with open(ML_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ML_FEATURES_PATH, "r") as f:
            features = json.load(f)
        print(f"[INFO] Loader ML-model og feature-liste: {features}")
        return model, features
    else:
        print("[ADVARSEL] Ingen trænet ML-model fundet – bruger random baseline.")
        return None, None

def reconcile_features(df, feature_list):
    missing = [col for col in feature_list if col not in df.columns]
    if missing:
        print(f"‼️ ADVARSEL: Følgende features manglede i data og blev tilføjet med 0: {missing}")
        for col in missing:
            df[col] = 0.0
    df = df[feature_list]
    return df

def load_pytorch_model(feature_dim, model_path=PYTORCH_MODEL_PATH, device_str="cpu"):
    if not os.path.exists(model_path):
        print(f"❌ PyTorch-model ikke fundet: {model_path}")
        return None
    model = TradingNet(input_dim=feature_dim, output_dim=2)
    model.load_state_dict(torch.load(model_path, map_location=device_str))
    model.eval()
    model.to(device_str)
    print(f"✅ PyTorch-model indlæst fra {model_path} på {device_str}")
    return model

def pytorch_predict(model, X, device_str="cpu"):
    X_ = X.copy()
    if "regime" in X_.columns and not np.issubdtype(X_["regime"].dtype, np.number):
        regime_map = {"bull": 1, "neutral": 0, "bear": -1}
        X_["regime"] = X_["regime"].map(regime_map).fillna(0)
    X_ = X_.apply(pd.to_numeric, errors="coerce").fillna(0)
    with torch.no_grad():
        X_tensor = torch.tensor(X_.values, dtype=torch.float32).to(device_str)
        logits = model(X_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def keras_lstm_predict(df, feature_cols, seq_length=48, model_path=LSTM_MODEL_PATH):
    from tensorflow.keras.models import load_model
    if not os.path.exists(model_path):
        print(f"❌ Keras LSTM-model ikke fundet: {model_path}")
        return np.zeros(len(df))
    mean = np.load(LSTM_SCALER_MEAN_PATH)
    scale = np.load(LSTM_SCALER_SCALE_PATH)
    df_X = reconcile_features(df, feature_cols)
    X = df_X.values
    X_scaled = (X - mean) / scale
    X_seq = []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i+seq_length])
    X_seq = np.array(X_seq)
    model = load_model(model_path)
    probs = model.predict(X_seq)
    preds = np.argmax(probs, axis=1)
    preds_full = np.zeros(len(df), dtype=int)
    preds_full[seq_length:] = preds
    return preds_full

def read_features_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    return df

GRAPH_DIR = "graphs/"
DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 1.0, 0.7]

def main(
    features_path,
    symbol="BTCUSDT",
    interval="1h",
    threshold=DEFAULT_THRESHOLD,
    weights=DEFAULT_WEIGHTS,
    device_str=None,
    use_lstm=False,
    FORCE_DEBUG=False
):
    device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    device_info = log_device_status(
        context="engine",
        extra={
            "symbol": symbol,
            "interval": interval,
            "model_type": "multi_compare"
        },
        telegram_func=send_message,
        print_console=True
    )

    monitor = ResourceMonitor(
        ram_max=85, cpu_max=90, gpu_max=95, gpu_temp_max=80, check_interval=10,
        action="pause", log_file="outputs/debug/resource_log.csv"
    )
    monitor.start()

    tb_run_name = f"engine_inference_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{tb_run_name}")

    try:
        print("🔄 Indlæser features:", features_path)
        df = read_features_auto(features_path)
        print(f"✅ Data indlæst ({len(df)} rækker)")
        print("Kolonner:", list(df.columns))

        # ---- ML (Random Forest, LightGBM etc.) ----
        print("🛠️ Loader ML-model ...")
        ml_model, ml_features = load_ml_model()
        if ml_model is not None and ml_features is not None:
            X_ml = reconcile_features(df, ml_features)
            ml_signals = ml_model.predict(X_ml)
        else:
            ml_signals = np.random.choice([0, 1], size=len(df))
            print("[ADVARSEL] ML fallback: bruger random signaler.")
        df["signal_ml"] = ml_signals
        trades_ml, balance_ml = run_backtest(df, signals=ml_signals)
        metrics_ml = calc_backtest_metrics(trades_ml, balance_ml)
        metrics_dict = {"ML": metrics_ml}

        # ---- DL (PyTorch eller LSTM) ----
        trained_features = load_trained_feature_list()
        if trained_features is not None:
            X_dl = reconcile_features(df, trained_features)
        else:
            fallback_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ("timestamp", "target", "regime", "signal")]
            X_dl = df[fallback_cols]
            print(f"[ADVARSEL] Kører med fallback-features: {fallback_cols}")

        print("🔄 Loader DL-model ...")
        if use_lstm and os.path.exists(LSTM_MODEL_PATH):
            print("✅ Bruger Keras LSTM til inference.")
            dl_signals = keras_lstm_predict(df, trained_features, seq_length=48, model_path=LSTM_MODEL_PATH)
            dl_probas = np.stack([1-dl_signals, dl_signals], axis=1)
        else:
            model = load_pytorch_model(feature_dim=X_dl.shape[1], device_str=device_str)
            if model is not None:
                dl_preds, dl_probas = pytorch_predict(model, X_dl, device_str=device_str)
                dl_signals = (dl_probas[:, 1] > threshold).astype(int)
                print("✅ PyTorch DL-inference klar!")
            else:
                print("❌ Ingen DL-model fundet – fallback til random signaler")
                dl_signals = np.random.choice([0, 1], size=len(df))
                dl_probas = np.stack([1-dl_signals, dl_signals], axis=1)
        df["signal_dl"] = dl_signals
        trades_dl, balance_dl = run_backtest(df, signals=dl_signals)
        metrics_dl = calc_backtest_metrics(trades_dl, balance_dl)
        metrics_dict["DL"] = metrics_dl

        # --- Ensemble ---
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

        print("\n=== Signal distributions ===")
        print("ML:", pd.Series(ml_signals).value_counts().to_dict())
        print("DL:", pd.Series(dl_signals).value_counts().to_dict())
        print("RSI:", pd.Series(rsi_signals).value_counts().to_dict())
        print("Ensemble:", pd.Series(ensemble_signals).value_counts().to_dict())

        print("\n=== Performance metrics (backtest) ===")
        for model, metrics in metrics_dict.items():
            print(f"{model}: {metrics}")

        # Telegram ensemble performance direkte
        send_ensemble_metrics(
            {
                "ml_test_acc": metrics_ml.get("profit_pct", 0)/100,
                "ml_train_acc": metrics_ml.get("profit_pct", 0)/100,  # evt. justér til rigtige værdier
                "ml_val_acc": metrics_ml.get("profit_pct", 0)/100,    # evt. justér
                "dl_test_acc": metrics_dl.get("profit_pct", 0)/100,
                "ensemble_test_acc": metrics_ens.get("profit_pct", 0)/100
            }
        )

        for model_name, metrics in metrics_dict.items():
            for metric_key, value in metrics.items():
                writer.add_scalar(f"{model_name}/{metric_key}", value)
        writer.flush()

        # --- Visualisering ---
        from visualization.plot_performance import plot_performance
        os.makedirs(GRAPH_DIR, exist_ok=True)
        plot_performance(balance_ml, trades_ml, model_name="ML", save_path=f"{GRAPH_DIR}/performance_ml.png")
        plot_performance(balance_dl, trades_dl, model_name="DL", save_path=f"{GRAPH_DIR}/performance_dl.png")
        plot_performance(balance_ens, trades_ens, model_name="Ensemble", save_path=f"{GRAPH_DIR}/performance_ensemble.png")

        from visualization.plot_comparison import plot_comparison
        metric_keys = ["profit_pct", "win_rate", "drawdown_pct", "num_trades"]
        plot_comparison(metrics_dict, metric_keys=metric_keys, save_path=f"{GRAPH_DIR}/model_comparison.png")
        print(f"[INFO] Sammenlignings-graf gemt til {GRAPH_DIR}/model_comparison.png")

        try:
            send_image(f"{GRAPH_DIR}/model_comparison.png", caption="ML vs. DL vs. ENSEMBLE performance")
        except Exception as e:
            print(f"[ADVARSEL] Telegram-graf kunne ikke sendes: {e}")

        print("\n🎉 Pipeline afsluttet uden fejl!")

    finally:
        writer.close()
        monitor.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Trading Engine")
    parser.add_argument("--features", type=str, required=True, help="Sti til feature-fil (CSV)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1h", help="Tidsinterval (fx 1h, 4h)")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device ('cuda'/'cpu'), auto hvis None")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for DL-signal")
    parser.add_argument("--weights", type=float, nargs=3, default=DEFAULT_WEIGHTS, help="Voting weights ML DL Rule")
    parser.add_argument("--use_lstm", action="store_true", help="Brug Keras LSTM-model i stedet for PyTorch (DL)")
    args = parser.parse_args()

    safe_run(lambda: main(
        features_path=args.features,
        symbol=args.symbol,
        interval=args.interval,
        threshold=args.threshold,
        weights=args.weights,
        device_str=args.device,
        use_lstm=args.use_lstm,
        FORCE_DEBUG=False
    ))
