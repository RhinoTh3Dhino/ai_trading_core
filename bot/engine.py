# -*- coding: utf-8 -*-
"""
AI Trading Engine – robust og failsafe.

- Sikre fallbacks for valgfri moduler (torch, strategier, visualisering, telegram, metrics).
- Deterministisk run_pipeline(...) til E2E-tests (OHLCV CSV -> simple signaler -> backtest -> outputs + backup-kopi).
- Holder din produktions-flow struktur intakt, men crasher ikke, hvis noget mangler.
"""
from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# --- PROJECT_ROOT (fallback hvis utils.project_path ikke findes) ---
try:
    from utils.project_path import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Telegram (failsafe) ---
try:
    from utils.telegram_utils import send_message, send_image
except Exception:
    def send_message(*args, **kwargs):
        return None
    def send_image(*args, **kwargs):
        return None

# --- Device logging (failsafe) ---
try:
    from utils.log_utils import log_device_status
except Exception:
    def log_device_status(*args, **kwargs):
        return {"device": "unknown"}

# --- SummaryWriter (failsafe) ---
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    class SummaryWriter:  # no-op fallback
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def flush(self): pass
        def close(self): pass

# --- Ressource monitor (failsafe) ---
try:
    from bot.monitor import ResourceMonitor  # type: ignore
except Exception:
    class ResourceMonitor:  # no-op fallback
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass

# --- Live metrics (failsafe) ---
try:
    from utils.monitoring_utils import send_live_metrics
except Exception:
    def send_live_metrics(*args, **kwargs):
        return None

# --- Konfig (failsafe) ---
try:
    from config.monitoring_config import (  # type: ignore
        ALARM_THRESHOLDS,
        ALERT_ON_DRAWNDOWN,  # bevarer dit eksisterende navn
        ALERT_ON_WINRATE,
        ALERT_ON_PROFIT,
        ENABLE_MONITORING,
    )
except Exception:
    ALARM_THRESHOLDS = {"drawdown": -20, "winrate": 0.20, "profit": -10}
    ALERT_ON_DRAWNDOWN = True
    ALERT_ON_WINRATE = True
    ALERT_ON_PROFIT = True
    ENABLE_MONITORING = True

# --- Versions (failsafe) ---
try:
    from versions import (  # type: ignore
        PIPELINE_VERSION, PIPELINE_COMMIT, FEATURE_VERSION,
        ENGINE_VERSION, ENGINE_COMMIT, MODEL_VERSION, LABEL_STRATEGY,
    )
except Exception:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = (
        ENGINE_COMMIT
    ) = MODEL_VERSION = LABEL_STRATEGY = "unknown"

# --- PyTorch (failsafe) ---
try:
    import torch  # type: ignore
except Exception:
    torch = None  # noqa: N816

# --- Backtest (failsafe stub) ---
try:
    from backtest.backtest import run_backtest  # type: ignore
except Exception:
    def run_backtest(df: pd.DataFrame, signals: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Minimal backtest-stub: åbn/luk trades ved signalændringer; naive balance."""
        df = df.copy()
        df["signal"] = signals.astype(int)
        df["price"] = df["close"].astype(float)
        entries = []
        position = 0
        balance = 1000.0
        equity = []
        for i in range(len(df)):
            sig = int(df["signal"].iat[i])
            price = float(df["price"].iat[i])
            # åbn/luk når sig ændres
            if sig == 1 and position == 0:
                position = 1
                entries.append({"idx": i, "type": "OPEN", "price": price})
            elif sig == 0 and position == 1:
                position = 0
                entries.append({"idx": i, "type": "CLOSE", "price": price})
            # pseudo equity
            equity.append(balance + (price - df["price"].iat[entries[-1]["idx"]]) if (position and entries) else balance)
        trades = pd.DataFrame(entries) if entries else pd.DataFrame(columns=["idx", "type", "price"])
        balance_df = pd.DataFrame({"balance": equity})
        return trades, balance_df

# --- Ensemble predict (failsafe) ---
try:
    from ensemble.ensemble_predict import ensemble_predict  # type: ignore
except Exception:
    def ensemble_predict(ml_preds, dl_preds, rule_preds, weights=None, voting="majority", debug=False):
        ml = np.asarray(ml_preds).astype(int)
        dl = np.asarray(dl_preds).astype(int)
        rl = np.asarray(rule_preds).astype(int)
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        w = np.asarray(weights, dtype=float)
        mat = np.vstack([ml, dl, rl]).T
        scores = mat @ w[:3]
        # majority-weighted
        thr = 0.5 * w[:3].sum()
        out = (scores >= thr).astype(int)
        return out

# --- Strategier (failsafe RSI) ---
try:
    from strategies.rsi_strategy import rsi_rule_based_signals  # type: ignore
except Exception:
    def rsi_rule_based_signals(df: pd.DataFrame, low: int = 45, high: int = 55) -> np.ndarray:
        """Meget simpel fallback: EMA10-kryds som proxy for RSI-neutral zone."""
        ema = df["close"].ewm(span=10, adjust=False).mean()
        return (df["close"] > ema).astype(int).to_numpy()

# (macd/ema_cross er ikke strengt nødvendige i denne engine; udelades hvis ikke findes)
try:
    from strategies.macd_strategy import macd_cross_signals  # type: ignore
except Exception:
    def macd_cross_signals(df: pd.DataFrame) -> np.ndarray:
        return rsi_rule_based_signals(df)

try:
    from strategies.ema_cross_strategy import ema_cross_signals  # type: ignore
except Exception:
    def ema_cross_signals(df: pd.DataFrame) -> np.ndarray:
        return rsi_rule_based_signals(df)

# --- Robust utils (failsafe) ---
try:
    from utils.robust_utils import safe_run  # type: ignore
except Exception:
    def safe_run(fn):
        try:
            return fn()
        except Exception as e:
            print(f"[safe_run] Fejl: {e}")
            return None

# =========================
# PRODUKTIONS-KONSTANTER
# =========================
MODEL_DIR = "models"
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
PYTORCH_FEATURES_PATH = os.path.join(MODEL_DIR, "best_pytorch_features.json")
LSTM_FEATURES_PATH = os.path.join(MODEL_DIR, "lstm_features.csv")
LSTM_SCALER_MEAN_PATH = os.path.join(MODEL_DIR, "lstm_scaler_mean.npy")
LSTM_SCALER_SCALE_PATH = os.path.join(MODEL_DIR, "lstm_scaler_scale.npy")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
ML_MODEL_PATH = os.path.join(MODEL_DIR, "best_ml_model.pkl")
ML_FEATURES_PATH = os.path.join(MODEL_DIR, "best_ml_features.json")

GRAPH_DIR = "graphs"
DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 1.0, 0.7]

# =========================
# HJÆLPEFUNKTIONER
# =========================
def load_trained_feature_list() -> Optional[List[str]]:
    if os.path.exists(PYTORCH_FEATURES_PATH):
        with open(PYTORCH_FEATURES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f)
        print(f"[INFO] Loader PyTorch feature-liste fra model: {features}")
        return features
    elif os.path.exists(LSTM_FEATURES_PATH):
        features = pd.read_csv(LSTM_FEATURES_PATH, header=None)[0].tolist()
        print(f"[INFO] Loader LSTM feature-liste fra model: {features}")
        return features
    else:
        print("[ADVARSEL] Ingen feature-liste fundet – bruger alle numeriske features fra input.")
        return None


def load_ml_model():
    if os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_FEATURES_PATH):
        with open(ML_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ML_FEATURES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f)
        print(f"[INFO] Loader ML-model og feature-liste: {features}")
        return model, features
    else:
        print("[ADVARSEL] Ingen trænet ML-model fundet – bruger random baseline.")
        return None, None


def reconcile_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_list if col not in df.columns]
    if missing:
        print(f"‼️ ADVARSEL: Følgende features manglede i data og blev tilføjet med 0: {missing}")
        for col in missing:
            df[col] = 0.0
    return df[feature_list]


def load_pytorch_model(feature_dim: int, model_path: str = PYTORCH_MODEL_PATH, device_str: str = "cpu"):
    if torch is None:
        print("❌ PyTorch ikke tilgængelig – springer DL over.")
        return None
    if not os.path.exists(model_path):
        print(f"❌ PyTorch-model ikke fundet: {model_path}")
        return None

    class TradingNet(torch.nn.Module):  # type: ignore
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

    model = TradingNet(input_dim=feature_dim, output_dim=2)
    model.load_state_dict(torch.load(model_path, map_location=device_str))  # type: ignore
    model.eval()
    model.to(device_str)  # type: ignore
    print(f"✅ PyTorch-model indlæst fra {model_path} på {device_str}")
    return model


def pytorch_predict(model, X: pd.DataFrame, device_str: str = "cpu"):
    X_ = X.copy()
    if "regime" in X_.columns and not np.issubdtype(X_["regime"].dtype, np.number):
        regime_map = {"bull": 1, "neutral": 0, "bear": -1}
        X_["regime"] = X_["regime"].map(regime_map).fillna(0)
    X_ = X_.apply(pd.to_numeric, errors="coerce").fillna(0)
    with torch.no_grad():  # type: ignore
        X_tensor = torch.tensor(X_.values, dtype=torch.float32).to(device_str)  # type: ignore
        logits = model(X_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()  # type: ignore
        preds = np.argmax(probs, axis=1)
    return preds, probs


def keras_lstm_predict(df: pd.DataFrame, feature_cols: List[str], seq_length: int = 48, model_path: str = LSTM_MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception:
        print("❌ TensorFlow/Keras ikke tilgængelig.")
        return np.zeros(len(df), dtype=int)

    if not os.path.exists(model_path):
        print(f"❌ Keras LSTM-model ikke fundet: {model_path}")
        return np.zeros(len(df), dtype=int)

    if not (os.path.exists(LSTM_SCALER_MEAN_PATH) and os.path.exists(LSTM_SCALER_SCALE_PATH)):
        print("❌ Mangler scaler-filer til LSTM – bruger nuller.")
        return np.zeros(len(df), dtype=int)

    mean = np.load(LSTM_SCALER_MEAN_PATH)
    scale = np.load(LSTM_SCALER_SCALE_PATH)
    df_X = reconcile_features(df, feature_cols)
    X = df_X.values
    X_scaled = (X - mean) / np.where(scale == 0, 1.0, scale)
    X_seq = []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i: i + seq_length])
    X_seq = np.array(X_seq)
    model = load_model(model_path)
    probs = model.predict(X_seq, verbose=0)
    preds = np.argmax(probs, axis=1)
    preds_full = np.zeros(len(df), dtype=int)
    preds_full[seq_length:] = preds
    return preds_full


def read_features_auto(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if str(first_line).startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    return df

# ============================================================
# TEST-ENTRYPOINT – deterministisk E2E
# ============================================================
def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _simple_signals(df: pd.DataFrame) -> np.ndarray:
    ema = df["close"].ewm(span=10, adjust=False).mean()
    return (df["close"] > ema).astype(int).to_numpy()


def run_pipeline(
    data_path: str,
    outputs_dir: str,
    backups_dir: str,
    paper: bool = True,
) -> Dict[str, float]:
    """
    Minimal, deterministisk pipeline til E2E-test:
      - Læser OHLCV CSV (kræver: timestamp|datetime, open, high, low, close, volume)
      - Genererer simple signaler
      - Kører backtest
      - Skriver outputs/signals.csv + outputs/portfolio_metrics.json
      - Laver backup_<ts>/ med kopi af metrics
    Returnerer metrics-dict.
    """
    outputs = Path(outputs_dir); outputs.mkdir(parents=True, exist_ok=True)
    backups = Path(backups_dir); backups.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV skal indeholde 'timestamp' eller 'datetime' kolonne.")

    df["timestamp"] = _ensure_datetime(df["timestamp"])
    if df["timestamp"].isna().any():
        raise ValueError("Kunne ikke parse nogle timestamps i input CSV.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i input CSV: {sorted(missing)}")

    df["ema_200"] = df["close"].ewm(span=10, adjust=False).mean()
    signals = _simple_signals(df)

    trades, balance = run_backtest(df, signals=signals)

    # metrics
    try:
        from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore
        metrics = _apm(trades, balance)
        if "max_drawdown" in metrics and "drawdown_pct" not in metrics:
            metrics["drawdown_pct"] = metrics["max_drawdown"]
    except Exception:
        pnl = float(
            (balance["balance"].iloc[-1] - balance["balance"].iloc[0])
            / max(balance["balance"].iloc[0], 1.0)
        ) if "balance" in balance else 0.0
        metrics = {
            "profit_pct": pnl * 100.0,
            "drawdown_pct": float(balance.get("drawdown", pd.Series([0])).min()) if "drawdown" in balance else 0.0,
            "num_trades": int((trades["type"] == "OPEN").sum()) if "type" in trades else 0,
        }

    # outputs
    sig_df = pd.DataFrame({"timestamp": df["timestamp"], "signal": signals.astype(int)})
    sig_path = outputs / "signals.csv"; sig_df.to_csv(sig_path, index=False)

    metrics_path = outputs / "portfolio_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ts_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bdir = backups / f"backup_{ts_name}"
    bdir.mkdir(parents=True, exist_ok=True)
    with (bdir / "portfolio_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[E2E] Skrev {sig_path} og {metrics_path}. Backup: {bdir}")
    return metrics

# ============================================================
# PRODUKTIONS-HOVEDFLOW
# ============================================================
def main(
    features_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    threshold: float = DEFAULT_THRESHOLD,
    weights: List[float] = DEFAULT_WEIGHTS,
    device_str: Optional[str] = None,
    use_lstm: bool = False,
    FORCE_DEBUG: bool = False,
) -> None:
    device_str = device_str or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
    device_info = log_device_status(
        context="engine",
        extra={"symbol": symbol, "interval": interval, "model_type": "multi_compare"},
        telegram_func=send_message,
        print_console=True,
    )

    monitor = ResourceMonitor(
        ram_max=85, cpu_max=90, gpu_max=95, gpu_temp_max=80,
        check_interval=10, action="pause",
        log_file=PROJECT_ROOT / "outputs" / "debug" / "resource_log.csv",
    )
    monitor.start()

    tb_run_name = f"engine_inference_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{tb_run_name}")

    try:
        print("🔄 Indlæser features:", features_path)
        df = read_features_auto(features_path)
        print(f"✅ Data indlæst ({len(df)} rækker)")
        print("Kolonner:", list(df.columns))

        # ---- ML ----
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
        try:
            from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore
            metrics_ml = _apm(trades_ml, balance_ml)
        except Exception:
            metrics_ml = {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0}
        metrics_dict: Dict[str, Dict[str, float]] = {"ML": metrics_ml}

        # ---- DL (PyTorch eller LSTM) ----
        trained_features = load_trained_feature_list()
        if trained_features is not None:
            X_dl = reconcile_features(df, trained_features)
        else:
            fallback_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in ("timestamp", "target", "regime", "signal")
            ]
            X_dl = df[fallback_cols]
            print(f"[ADVARSEL] Kører med fallback-features: {fallback_cols}")

        print("🔄 Loader DL-model ...")
        if use_lstm and os.path.exists(LSTM_MODEL_PATH):
            print("✅ Bruger Keras LSTM til inference.")
            dl_signals = keras_lstm_predict(df, trained_features, seq_length=48, model_path=LSTM_MODEL_PATH)
            dl_probas = np.stack([1 - dl_signals, dl_signals], axis=1)
        else:
            model = load_pytorch_model(feature_dim=X_dl.shape[1], device_str=device_str)
            if model is not None and torch is not None:
                dl_preds, dl_probas = pytorch_predict(model, X_dl, device_str=device_str)
                dl_signals = (dl_probas[:, 1] > threshold).astype(int)
                print("✅ PyTorch DL-inference klar!")
            else:
                print("❌ Ingen DL-model fundet – fallback til random signaler")
                dl_signals = np.random.choice([0, 1], size=len(df))
                dl_probas = np.stack([1 - dl_signals, dl_signals], axis=1)
        df["signal_dl"] = dl_signals

        trades_dl, balance_dl = run_backtest(df, signals=dl_signals)
        try:
            from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore
            metrics_dl = _apm(trades_dl, balance_dl)
        except Exception:
            metrics_dl = {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0}
        metrics_dict["DL"] = metrics_dl

        # --- Ensemble ---
        rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
        ensemble_signals = ensemble_predict(
            ml_preds=ml_signals,
            dl_preds=dl_signals,
            rule_preds=rsi_signals,
            weights=DEFAULT_WEIGHTS,
            voting="majority",
            debug=True,
        )
        df["signal_ensemble"] = ensemble_signals

        trades_ens, balance_ens = run_backtest(df, signals=ensemble_signals)
        try:
            from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore
            metrics_ens = _apm(trades_ens, balance_ens)
        except Exception:
            metrics_ens = {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0}
        metrics_dict["Ensemble"] = metrics_ens

        print("\n=== Signal distributions ===")
        print("ML:", pd.Series(ml_signals).value_counts().to_dict())
        print("DL:", pd.Series(dl_signals).value_counts().to_dict())
        print("RSI:", pd.Series(rsi_signals).value_counts().to_dict())
        print("Ensemble:", pd.Series(ensemble_signals).value_counts().to_dict())

        print("\n=== Performance metrics (backtest) ===")
        for model, metrics in metrics_dict.items():
            print(f"{model}: {metrics}")

        # === Live monitoring/alerting ===
        if ENABLE_MONITORING:
            send_live_metrics(
                trades_ens,
                balance_ens,
                symbol=symbol,
                timeframe=interval,
                thresholds=ALARM_THRESHOLDS,
                alert_on_drawdown=ALERT_ON_DRAWNDOWN,
                alert_on_winrate=ALERT_ON_WINRATE,
                alert_on_profit=ALERT_ON_PROFIT,
            )

        # Logging til TensorBoard
        for model_name, metrics in metrics_dict.items():
            for metric_key, value in metrics.items():
                try:
                    writer.add_scalar(f"{model_name}/{metric_key}", float(value))
                except Exception:
                    pass
        writer.flush()

        # --- Visualisering (failsafe no-op hvis modul mangler) ---
        try:
            from visualization.plot_performance import plot_performance  # type: ignore
            os.makedirs(GRAPH_DIR, exist_ok=True)
            plot_performance(balance_ml, trades_ml, model_name="ML", save_path=f"{GRAPH_DIR}/performance_ml.png")
            plot_performance(balance_dl, trades_dl, model_name="DL", save_path=f"{GRAPH_DIR}/performance_dl.png")
            plot_performance(balance_ens, trades_ens, model_name="Ensemble", save_path=f"{GRAPH_DIR}/performance_ensemble.png")
        except Exception as e:
            print(f"[ADVARSEL] plot_performance mangler/fejlede: {e}")

        try:
            from visualization.plot_comparison import plot_comparison  # type: ignore
            metric_keys = ["profit_pct", "max_drawdown", "sharpe", "sortino"]
            os.makedirs(GRAPH_DIR, exist_ok=True)
            plot_comparison(metrics_dict, metric_keys=metric_keys, save_path=f"{GRAPH_DIR}/model_comparison.png")
            print(f"[INFO] Sammenlignings-graf gemt til {GRAPH_DIR}/model_comparison.png")
            try:
                send_image(f"{GRAPH_DIR}/model_comparison.png", caption="ML vs. DL vs. ENSEMBLE performance")
            except Exception as e:
                print(f"[ADVARSEL] Telegram-graf kunne ikke sendes: {e}")
        except Exception as e:
            print(f"[ADVARSEL] plot_comparison mangler/fejlede: {e}")

        print("\n🎉 Pipeline afsluttet uden fejl!")

    finally:
        try:
            writer.close()
        except Exception:
            pass
        try:
            monitor.stop()
        except Exception:
            pass

# =========================
# CLI
# =========================
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

    safe_run(
        lambda: main(
            features_path=args.features,
            symbol=args.symbol,
            interval=args.interval,
            threshold=args.threshold,
            weights=list(args.weights) if isinstance(args.weights, (list, tuple)) else DEFAULT_WEIGHTS,
            device_str=args.device,
            use_lstm=args.use_lstm,
            FORCE_DEBUG=False,
        )
    )
