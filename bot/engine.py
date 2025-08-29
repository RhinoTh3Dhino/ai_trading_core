# -*- coding: utf-8 -*-
"""
AI Trading Engine – robust, failsafe og nu med PAPER TRADING mode.

NYT:
- --mode paper: PaperBroker integration, bar-for-bar handel, equity/fills CSV, daglig metrics-logging.
- Daglig aggregering → logs/daily_metrics.csv (win-rate, signal_count, trades, gross/net PnL, max_dd, sharpe_d approx).
- Valgfri Telegram daglig rapport.
- AUTO features: --features auto (default) genererer/loader {SYMBOL}_{TF}_latest.csv uden hårdkodning.
- ✅ Robust udlæsning af aktuel position (qty) så SELL/close virker, uanset brokerens datastruktur.

Bevarer:
- Analyse/backtest-flow (ML/DL/Ensemble) og al tidligere robusthed.
"""
from __future__ import annotations

import os
import csv
import json
import math
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# --- PROJECT_ROOT (fallback hvis utils.project_path ikke findes) ---
try:
    from utils.project_path import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- .env (vigtigt: load fra projektroden) ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

LOGS_DIR = Path(PROJECT_ROOT) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DAILY_METRICS_CSV = LOGS_DIR / "daily_metrics.csv"
EQUITY_CSV = LOGS_DIR / "equity.csv"
FILLS_CSV = LOGS_DIR / "fills.csv"
SIGNALS_CSV = LOGS_DIR / "signals.csv"  # NY: bar-for-bar signal logging

# --- Telegram (failsafe) ---
try:
    from utils.telegram_utils import send_message, send_image, send_live_metrics  # type: ignore
except Exception:
    def send_message(*args, **kwargs):
        return None
    def send_image(*args, **kwargs):
        return None
    def send_live_metrics(*args, **kwargs):
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

# --- Konfig (failsafe) ---
try:
    from config.monitoring_config import (  # type: ignore
        ALARM_THRESHOLDS,
        ALERT_ON_DRAWNDOWN,
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
        df = df.copy()
        df["signal"] = signals.astype(int)
        df["price"] = df["close"].astype(float)
        entries = []
        position = 0
        balance = 1000.0
        equity = []
        entry_idx = None
        for i in range(len(df)):
            sig = int(df["signal"].iat[i])
            price = float(df["price"].iat[i])
            if sig == 1 and position == 0:
                position = 1
                entry_idx = i
                entries.append({"idx": i, "type": "OPEN", "price": price})
            elif sig == 0 and position == 1:
                position = 0
                entries.append({"idx": i, "type": "CLOSE", "price": price})
                entry_idx = None
            if position and entry_idx is not None:
                equity.append(balance + (price - df["price"].iat[entry_idx]))
            else:
                equity.append(balance)
        trades = pd.DataFrame(entries) if entries else pd.DataFrame(columns=["idx", "type", "price"])
        balance = pd.DataFrame({"balance": equity})
        return trades, balance

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
        thr = 0.5 * w[:3].sum()
        out = (scores >= thr).astype(int)
        return out

# --- Strategier (failsafe RSI) ---
try:
    from strategies.rsi_strategy import rsi_rule_based_signals  # type: ignore
except Exception:
    def rsi_rule_based_signals(df: pd.DataFrame, low: int = 45, high: int = 55) -> np.ndarray:
        ema = df["close"].ewm(span=10, adjust=False).mean()
        return (df["close"] > ema).astype(int).to_numpy()

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

# --- PaperBroker (failsafe stub) ---
try:
    from bot.brokers.paper_broker import PaperBroker  # type: ignore
except Exception:
    class PaperBroker:  # minimal stub – anbefalet at bruge den rigtige fil
        def __init__(self, **k):
            self.cash = float(k.get("starting_cash", 100000))
            self.positions = {}
            self.realized_pnl = 0.0
            self.trading_halted = False
            self._last_px = {}
            self.equity_log_path = k.get("equity_log_path", None)
            self.fills_log_path = k.get("fills_log_path", None)
        def mark_to_market(self, prices: Dict[str, float], ts: Optional[datetime] = None):
            self._last_px.update(prices)
            equity = self.cash + sum(self.positions.get(s, 0)*p for s,p in prices.items())
            return {"equity": equity, "cash": self.cash, "positions_value": 0.0, "drawdown_pct": 0.0,
                    "positions": {}, "open_orders": [], "trading_halted": self.trading_halted}
        def submit_order(self, symbol, side, qty, order_type="market", ts: Optional[datetime]=None, **k):
            px = self._last_px.get(symbol)
            if px is None: return {"status":"rejected"}
            notional = px * qty
            if side=="BUY": self.cash -= notional
            else: self.cash += notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) + (qty if side=="BUY" else -qty)
            return {"status":"filled"}
        def pnl_snapshot(self, prices=None):
            equity = self.cash + sum(self.positions.get(s,0)*p for s,p in (prices or {}).items())
            return {"realized_pnl":0.0, "unrealized_pnl":0.0, "equity":equity, "cash":self.cash}
        def close_position(self, symbol: str, ts: Optional[datetime]=None):
            qty = self.positions.get(symbol, 0.0)
            if qty != 0:
                side = "SELL" if qty>0 else "BUY"
                self.submit_order(symbol, side, abs(qty), order_type="market", ts=ts)

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

# --- AUTO features (NY) ---
try:
    from features.auto_features import ensure_latest  # type: ignore
except Exception:
    # Fallback: find {symbol}_{tf}_latest.csv i outputs/feature_data eller tag nyeste CSV i mappen
    def ensure_latest(symbol: str = "BTCUSDT", timeframe: str = "1h", min_rows: int = 200):
        outdir = Path(PROJECT_ROOT) / "outputs" / "feature_data"
        outdir.mkdir(parents=True, exist_ok=True)
        candidate = outdir / f"{symbol}_{timeframe}_latest.csv"
        if candidate.exists():
            return candidate
        alts = sorted(outdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if alts:
            return alts[0]
        raise FileNotFoundError("ensure_latest fallback: ingen features-CSV fundet i outputs/feature_data/")

def _resolve_features_path(features_path: Optional[str], symbol: str, interval: str, *, min_rows: int = 200) -> str:
    """
    Returnér endelig sti til features CSV:
    - Hvis 'auto' eller ikke eksisterende → ensure_latest(symbol, interval)
    - Ellers returnér angivet sti uændret
    """
    if not features_path or str(features_path).lower() == "auto" or not os.path.exists(str(features_path)):
        path = ensure_latest(symbol=symbol, timeframe=interval, min_rows=min_rows)
        print(f"🧩 AUTO features valgt → {path}")
        return str(path)
    return str(features_path)

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
# ENV-HELPERS (defaults fra .env)
# =========================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)

ENV_SYMBOL = _env_str("SYMBOL", "BTCUSDT")
ENV_INTERVAL = _env_str("TIMEFRAME", "1h")
ENV_MODE = _env_str("MODE", "analyze")  # analyze|paper
ENV_FEATURES = _env_str("FEATURES", "auto")
ENV_DEVICE = os.getenv("PYTORCH_DEVICE")  # None => auto
ENV_ALLOC_PCT = _env_float("ALLOC_PCT", 0.10)
ENV_COMMISSION_BP = _env_float("COMMISSION_BP", 2.0)
ENV_SLIPPAGE_BP = _env_float("SLIPPAGE_BP", 1.0)
ENV_DAILY_LOSS_LIMIT_PCT = _env_float("DAILY_LOSS_LIMIT_PCT", 0.0)

# =========================
# HJÆLPEFUNKTIONER (fælles)
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


def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _simple_signals(df: pd.DataFrame) -> np.ndarray:
    ema = df["close"].ewm(span=10, adjust=False).mean()
    return (df["close"] > ema).astype(int).to_numpy()

# -------------------------------------------------------------
# Position-helpers (NY): robust udlæsning af nuværende qty
# -------------------------------------------------------------
def _extract_numeric(obj, keys):
    """Prøv at hive et numerisk felt ud fra dict/obj via en liste af kandidatfelter."""
    for k in keys:
        try:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                return float(obj[k])
            if hasattr(obj, k) and getattr(obj, k) is not None:
                return float(getattr(obj, k))
        except Exception:
            pass
    # én nesting ned hvis dict (fx {"BTCUSDT": {"position": {"qty": 0.1}}})
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict):
                val = _extract_numeric(v, keys)
                if val is not None:
                    return val
    return None

def _get_current_qty(broker, symbol: str) -> float:
    """Returnér nuværende position-qty for symbol uanset brokerens datastruktur."""
    pos_container = getattr(broker, "positions", None)
    if pos_container is None:
        return 0.0
    pos = pos_container.get(symbol) if isinstance(pos_container, dict) else pos_container
    if pos is None:
        return 0.0
    if isinstance(pos, (int, float)):
        return float(pos)
    val = _extract_numeric(
        pos,
        ["qty", "quantity", "size", "position", "position_size",
         "amount", "net", "net_qty", "positionQty", "position_qty"]
    )
    return float(val) if val is not None else 0.0

# ============================================================
# E2E TEST-PIPELINE (uændret)
# ============================================================
def run_pipeline(
    data_path: str,
    outputs_dir: str,
    backups_dir: str,
    paper: bool = True,
) -> Dict[str, float]:
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

    sig_df = pd.DataFrame({"timestamp": df["timestamp"], "signal": signals.astype(int)})
    sig_path = Path(outputs) / "signals.csv"; sig_df.to_csv(sig_path, index=False)

    metrics_path = Path(outputs) / "portfolio_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ts_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bdir = Path(backups) / f"backup_{ts_name}"
    bdir.mkdir(parents=True, exist_ok=True)
    with (bdir / "portfolio_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[E2E] Skrev {sig_path} og {metrics_path}. Backup: {bdir}")
    return metrics

# ============================================================
# PAPER TRADING – daglige metrikker & Telegram
# ============================================================
def _ensure_daily_metrics_headers() -> None:
    if not DAILY_METRICS_CSV.exists():
        with DAILY_METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "signal_count", "trades", "win_rate", "gross_pnl", "net_pnl", "max_dd", "sharpe_d"])

def _calc_daily_metrics_for_date(date_str: str) -> Dict[str, float]:
    """Læs fills/equity/signals for en bestemt dag og beregn metrikker.
    Fallback: hvis ingen lukkede handler (pnl_realized), brug equity-diff og 1->0 flips.
    """
    gross = 0.0
    wins = 0
    closed = 0
    commissions = 0.0

    # --- 1) Forsøg at læse realiseret PnL fra fills ---
    if FILLS_CSV.exists():
        df_f = pd.read_csv(FILLS_CSV)
        if "ts" in df_f.columns:
            df_f["date"] = df_f["ts"].astype(str).str[:10]
            d = df_f[df_f["date"] == date_str]
            if not d.empty:
                if "pnl_realized" in d.columns:
                    pnl_series = pd.to_numeric(d["pnl_realized"], errors="coerce").fillna(0.0)
                    gross = float(pnl_series.sum())
                    wins = int((pnl_series > 0).sum())
                    closed = int((pnl_series != 0).sum())
                if "commission" in d.columns:
                    commissions = float(pd.to_numeric(d["commission"], errors="coerce").fillna(0.0).sum())

    # --- 2) Equity-drevet fallback, hvis ingen lukkede handler ---
    if closed == 0 and EQUITY_CSV.exists():
        df_e = pd.read_csv(EQUITY_CSV)
        if {"date","equity"}.issubset(df_e.columns):
            day = df_e[df_e["date"].astype(str) == date_str]
            if not day.empty:
                e = pd.to_numeric(day["equity"], errors="coerce").dropna().values
                if e.size >= 2:
                    gross = float(e[-1] - e[0])  # PnL ~ dagens equity-forskel
                    # trades ≈ antal 1->0 flips i signals (lukke-signaler)
                    if SIGNALS_CSV.exists():
                        try:
                            s = pd.read_csv(SIGNALS_CSV)
                            if {"ts","signal"}.issubset(s.columns):
                                s["date"] = s["ts"].astype(str).str[:10]
                                d = s[s["date"] == date_str].copy()
                                d = d.sort_values("ts")
                                sig = pd.to_numeric(d["signal"], errors="coerce").fillna(0).astype(int).to_numpy()
                                closed = int(((sig[:-1] == 1) & (sig[1:] == 0)).sum())
                        except Exception:
                            pass
                    wins = 0  # kan ikke kendes uden trade-PnL

    net = gross - commissions

    # --- 3) Max intradag drawdown & "Sharpe_d" fra equity ---
    max_dd = 0.0
    sharpe_d = 0.0
    if EQUITY_CSV.exists():
        df_e = pd.read_csv(EQUITY_CSV)
        if {"date","equity"}.issubset(df_e.columns):
            day_e = df_e[df_e["date"].astype(str) == date_str]
            if not day_e.empty:
                e = pd.to_numeric(day_e["equity"], errors="coerce").dropna().values
                if e.size:
                    peak = -1e18
                    dd_pct = 0.0
                    for val in e:
                        peak = max(peak, val)
                        dd_pct = min(dd_pct, (val - peak) / (peak + 1e-12) * 100.0)
                    max_dd = dd_pct
                    rets = np.diff(e)
                    if rets.size > 1 and np.std(rets) > 1e-12:
                        sharpe_d = float(np.mean(rets) / np.std(rets))

    # --- 4) signal_count (skrives separat under løb) ---
    signal_count = 0
    if DAILY_METRICS_CSV.exists():
        dm = pd.read_csv(DAILY_METRICS_CSV)
        row = dm[dm["date"].astype(str) == date_str]
        if not row.empty and "signal_count" in row.columns:
            try:
                signal_count = int(row["signal_count"].iloc[0])
            except Exception:
                signal_count = 0

    win_rate = float(wins / max(closed, 1) * 100.0)
    return {
        "signal_count": signal_count,
        "trades": int(closed),
        "win_rate": round(win_rate, 2),
        "gross_pnl": round(gross, 2),
        "net_pnl": round(net, 2),
        "max_dd": round(max_dd, 2),
        "sharpe_d": round(sharpe_d, 2),
    }

def _upsert_daily_metrics(date_str: str, updates: Dict[str, float]) -> None:
    """Idempotent opdatering/indsættelse af en dags række."""
    _ensure_daily_metrics_headers()
    rows = []
    found = False
    if DAILY_METRICS_CSV.exists():
        with DAILY_METRICS_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["date"] == date_str:
                    r.update({k: str(v) for k, v in updates.items()})
                    found = True
                rows.append(r)
    if not found:
        base = {"date": date_str, "signal_count": "0", "trades": "0", "win_rate": "0", "gross_pnl": "0", "net_pnl": "0", "max_dd": "0", "sharpe_d": "0"}
        base.update({k: str(v) for k, v in updates.items()})
        rows.append(base)
    with DAILY_METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "signal_count", "trades", "win_rate", "gross_pnl", "net_pnl", "max_dd", "sharpe_d"])
        writer.writeheader()
        writer.writerows(rows)

def _send_daily_report_telegram(date_str: str, m: Dict[str, float]) -> None:
    try:
        msg = (
            f"📊 *Daglig rapport* {date_str}\n"
            f"- Win-rate: {m['win_rate']:.2f}%\n"
            f"- Signal count: {m['signal_count']}\n"
            f"- Trades (closed legs): {m['trades']}\n"
            f"- Gross PnL: {m['gross_pnl']:.2f}\n"
            f"- Net PnL: {m['net_pnl']:.2f}\n"
            f"- Max DD: {m['max_dd']:.2f}%\n"
            f"- Sharpe_d: {m['sharpe_d']:.2f}\n"
        )
        # Robust i telegram_utils: MarkdownV2 escapes + fallback
        send_message(msg, parse_mode="MarkdownV2")
    except Exception as e:
        print(f"[INFO] Telegram ikke konfigureret/fejlede: {e}")

def _append_signals_rows(rows: List[Dict[str, str | int]]) -> None:
    """Append liste af {'ts': ISO, 'signal': int} til logs/signals.csv med header ved behov."""
    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SIGNALS_CSV.exists()
    with SIGNALS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "signal"])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({"ts": r["ts"], "signal": r["signal"]})

# ============================================================
# PAPER TRADING – hovedløb
# ============================================================
def _generate_ensemble_signals_for_df(df: pd.DataFrame, threshold: float, device_str: str, use_lstm: bool) -> np.ndarray:
    """Genbrug eksisterende model-loading for at producere ensemble-signaler for hele DF."""
    # ML
    ml_model, ml_features = load_ml_model()
    if ml_model is not None and ml_features is not None:
        X_ml = reconcile_features(df.copy(), ml_features)
        ml_signals = ml_model.predict(X_ml)
    else:
        ml_signals = np.random.choice([0, 1], size=len(df))
        print("[ADVARSEL] ML fallback: random signaler.")

    # DL
    trained_features = load_trained_feature_list()
    if trained_features is not None:
        X_dl = reconcile_features(df.copy(), trained_features)
    else:
        fallback_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in ("timestamp", "target", "regime", "signal")
        ]
        X_dl = df[fallback_cols]
        print(f"[ADVARSEL] DL fallback-features: {fallback_cols}")

    # LSTM → PyTorch → Random, men: hvis --use_lstm og filer mangler, neutraliser DL (0)
    lstm_ok = bool(use_lstm) and all(
        os.path.exists(p) for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH)
    )
    if lstm_ok:
        feature_cols = trained_features if trained_features is not None else list(X_dl.columns)
        dl_signals = keras_lstm_predict(df, feature_cols, seq_length=48, model_path=LSTM_MODEL_PATH)
    elif use_lstm:
        missing = [p for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH) if not os.path.exists(p)]
        print(f"⚠️ --use_lstm er sat, men mangler filer: {missing}. Hopper DL over (neutral stemme).")
        dl_signals = np.zeros(len(df), dtype=int)
    else:
        model = load_pytorch_model(feature_dim=X_dl.shape[1], device_str=device_str)
        if model is not None and torch is not None:
            _, dl_probs = pytorch_predict(model, X_dl, device_str=device_str)
            dl_signals = (dl_probs[:, 1] > threshold).astype(int)
        else:
            print("❌ Ingen DL-model – random signaler.")
            dl_signals = np.random.choice([0, 1], size=len(df))

    # Rule (binariser >0 -> 1, ellers 0)
    rsi_signals_raw = rsi_rule_based_signals(df, low=45, high=55)
    rsi_signals = np.where(rsi_signals_raw > 0, 1, 0)

    # Ensemble
    ens = ensemble_predict(ml_signals, dl_signals, rsi_signals, weights=DEFAULT_WEIGHTS, voting="majority", debug=False)
    return ens.astype(int)

def run_paper_trading(
    features_path: str,
    symbol: str,
    interval: str,
    *,
    threshold: float,
    device_str: str,
    use_lstm: bool,
    commission_bp: float,
    slippage_bp: float,
    daily_loss_limit_pct: float,
    allow_short: bool,
    alloc_pct: float,
) -> None:
    """
    Paper trading: går bar-for-bar gennem features-CSV, genererer ensemble-signal og
    eksekverer ordre for at matche regime: signal=1 -> long, signal=0 -> flat (long-only).
    """
    # AUTO features (ingen hårdkodning)
    features_path = _resolve_features_path(features_path, symbol, interval, min_rows=200)

    print(f"🔄 Indlæser features til paper: {features_path}")
    df = read_features_auto(features_path)
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("Features skal have 'timestamp' eller 'datetime' kolonne.")
    df["timestamp"] = _ensure_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError("Features skal have 'close' kolonne.")

    # generér alle signaler på forhånd (samme som analyse-mode)
    ens_signals = _generate_ensemble_signals_for_df(df.copy(), threshold, device_str, use_lstm)
    df["signal_ens"] = ens_signals

    # init broker
    broker = PaperBroker(
        starting_cash=100_000.0,
        commission_bp=commission_bp,
        slippage_bp=slippage_bp,
        daily_loss_limit_pct=daily_loss_limit_pct,
        allow_short=allow_short,
        equity_log_path=EQUITY_CSV,
        fills_log_path=FILLS_CSV,
    )

    # daglige tællere
    current_day = None
    daily_signal_count = 0
    prev_sig = 0

    # buffer til signals.csv (bar-for-bar)
    signal_rows: List[Dict[str, str | int]] = []

    print("🚀 Starter bar-for-bar loop…")
    for i in range(len(df)):
        ts = df["timestamp"].iat[i]
        price = float(df["close"].iat[i])
        sig = int(df["signal_ens"].iat[i])

        # log signal til buffer (brugt af aggregator til signal_count via flips)
        signal_rows.append({"ts": ts.isoformat(), "signal": sig})

        # dato
        dstr = ts.strftime("%Y-%m-%d")
        if current_day is None:
            current_day = dstr

        # mark-to-market (opdater equity + fyld evt. åbne limitordrer)
        broker.mark_to_market({symbol: price}, ts=ts)

        # dagsskifte → opdater daglige metrikker (inkl. signal_count)
        if dstr != current_day:
            _upsert_daily_metrics(current_day, {"signal_count": int(daily_signal_count)})
            m = _calc_daily_metrics_for_date(current_day)
            _upsert_daily_metrics(current_day, m)
            _send_daily_report_telegram(current_day, m)
            daily_signal_count = 0
            current_day = dstr

        # signalflip tælling (kun 0→1 tælles som "entry-signal")
        if sig == 1 and prev_sig == 0:
            daily_signal_count += 1
        prev_sig = sig

        # eksekver strategi: målposition = long 100% af allokeret kapital ved sig=1, ellers flat
        snap = broker.pnl_snapshot({symbol: price})
        equity = float(snap.get("equity", 0.0))
        target_notional = equity * (alloc_pct if sig == 1 else 0.0)
        target_qty = round(target_notional / max(price, 1e-9), 8)

        # ✅ ROBUST: find nuværende qty
        cur_qty = _get_current_qty(broker, symbol)

        delta = target_qty - cur_qty
        if abs(delta) > 1e-8 and not getattr(broker, "trading_halted", False):
            side = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)
            ord_res = broker.submit_order(symbol, side, qty, order_type="market", ts=ts)
            status = ord_res.get("status") if isinstance(ord_res, dict) else getattr(ord_res, "status", "")
            if status == "rejected":
                if target_qty == 0.0 and cur_qty != 0.0:
                    if hasattr(broker, "close_position"):
                        broker.close_position(symbol, ts=ts)
                    else:
                        # simple fallback – modhandlen
                        broker.submit_order(symbol, "SELL" if cur_qty > 0 else "BUY", abs(cur_qty), order_type="market", ts=ts)

    # flush signaler til logs/signals.csv
    if signal_rows:
        _append_signals_rows(signal_rows)

    # efter sidste bar → afslut dag
    if current_day is not None:
        _upsert_daily_metrics(current_day, {"signal_count": int(daily_signal_count)})
        m = _calc_daily_metrics_for_date(current_day)
        _upsert_daily_metrics(current_day, m)
        _send_daily_report_telegram(current_day, m)

    print("✅ Paper trading gennemløb færdigt.")
    print(f"- Fills: {FILLS_CSV}")
    print(f"- Equity: {EQUITY_CSV}")
    print(f"- Daily metrics: {DAILY_METRICS_CSV}")
    print(f"- Signals: {SIGNALS_CSV}")

# ============================================================
# PRODUKTIONS-HOVEDFLOW (analyze-mode med AUTO features)
# ============================================================
def main(
    features_path: Optional[str] = ENV_FEATURES,
    symbol: str = ENV_SYMBOL,
    interval: str = ENV_INTERVAL,
    threshold: float = DEFAULT_THRESHOLD,
    weights: List[float] = DEFAULT_WEIGHTS,
    device_str: Optional[str] = ENV_DEVICE,
    use_lstm: bool = False,
    FORCE_DEBUG: bool = False,
    mode: str = ENV_MODE,            # "analyze" | "paper"
    commission_bp: float = ENV_COMMISSION_BP,
    slippage_bp: float = ENV_SLIPPAGE_BP,
    daily_loss_limit_pct: float = ENV_DAILY_LOSS_LIMIT_PCT,
    allow_short: bool = False,
    alloc_pct: float = ENV_ALLOC_PCT,  # andel af equity pr. entry
) -> None:
    device_str = device_str or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
    _ = log_device_status(
        context="engine",
        extra={"symbol": symbol, "interval": interval, "model_type": "multi_compare", "mode": mode},
        telegram_func=send_message,
        print_console=True,
    )

    if mode.lower() == "paper":
        run_paper_trading(
            features_path=features_path,  # resolve sker inde i run_paper_trading
            symbol=symbol,
            interval=interval,
            threshold=threshold,
            device_str=device_str,
            use_lstm=use_lstm,
            commission_bp=commission_bp,
            slippage_bp=slippage_bp,
            daily_loss_limit_pct=daily_loss_limit_pct,
            allow_short=allow_short,
            alloc_pct=alloc_pct,
        )
        return

    # ===== analyze-mode =====
    monitor = ResourceMonitor(
        ram_max=85, cpu_max=90, gpu_max=95, gpu_temp_max=80,
        check_interval=10, action="pause",
        log_file=PROJECT_ROOT / "outputs" / "debug" / "resource_log.csv",
    )
    monitor.start()

    tb_run_name = f"engine_inference_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{tb_run_name}")

    try:
        features_path_final = _resolve_features_path(features_path, symbol, interval, min_rows=200)
        print("🔄 Indlæser features:", features_path_final)
        df = read_features_auto(features_path_final)
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

        print(f"🔄 Loader DL-model ...{' (LSTM ønsket)' if use_lstm else ''}")
        lstm_ok = bool(use_lstm) and all(
            os.path.exists(p) for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH)
        )
        if lstm_ok:
            print("✅ Bruger Keras LSTM til inference.")
            feature_cols = trained_features if trained_features is not None else list(X_dl.columns)
            dl_signals = keras_lstm_predict(df, feature_cols, seq_length=48, model_path=LSTM_MODEL_PATH)
            dl_probas = np.stack([1 - dl_signals, dl_signals], axis=1, dtype=float)
        elif use_lstm:
            missing = [p for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH) if not os.path.exists(p)]
            print(f"⚠️ --use_lstm er sat, men mangler filer: {missing}. Hopper DL over (neutral stemme).")
            dl_signals = np.zeros(len(df), dtype=int)
            dl_probas = np.stack([1 - dl_signals, dl_signals], axis=1, dtype=float)
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

        # --- Ensemble (Rule → binær) ---
        rsi_signals_raw = rsi_rule_based_signals(df, low=45, high=55)
        rsi_signals = np.where(rsi_signals_raw > 0, 1, 0)
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
    parser.add_argument("--features", type=str, default=ENV_FEATURES, help="Sti til feature-fil (CSV) eller 'auto' (default)")
    parser.add_argument("--symbol", type=str, default=ENV_SYMBOL, help="Trading symbol")
    parser.add_argument("--interval", type=str, default=ENV_INTERVAL, help="Tidsinterval (fx 1h, 4h)")
    parser.add_argument("--device", type=str, default=ENV_DEVICE, help="PyTorch device ('cuda'/'cpu'), auto hvis None")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for DL-signal")
    parser.add_argument("--weights", type=float, nargs=3, default=DEFAULT_WEIGHTS, help="Voting weights ML DL Rule")
    parser.add_argument("--use_lstm", action="store_true", help="Brug Keras LSTM-model i stedet for PyTorch (DL)")

    # Paper-mode
    parser.add_argument("--mode", type=str, choices=["analyze", "paper"], default=ENV_MODE, help="Kørselstilstand")
    parser.add_argument("--commission-bp", type=float, default=ENV_COMMISSION_BP, dest="commission_bp", help="Kommission i basispoint (0.01% = 1bp)")
    parser.add_argument("--slippage-bp", type=float, default=ENV_SLIPPAGE_BP, dest="slippage_bp", help="Slippage i basispoint")
    parser.add_argument("--daily-loss-limit-pct", type=float, default=ENV_DAILY_LOSS_LIMIT_PCT, dest="daily_loss_limit_pct", help="Dagligt tab-stop i % (0 = off)")
    parser.add_argument("--allow-short", action="store_true", help="Tillad netto short i paper-mode")
    parser.add_argument("--alloc-pct", type=float, default=ENV_ALLOC_PCT, dest="alloc_pct", help="Andel af equity per entry (0.10 = 10%)")

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
            mode=args.mode,
            commission_bp=args.commission_bp,
            slippage_bp=args.slippage_bp,
            daily_loss_limit_pct=args.daily_loss_limit_pct,
            allow_short=args.allow_short,
            alloc_pct=args.alloc_pct,
        )
    )
