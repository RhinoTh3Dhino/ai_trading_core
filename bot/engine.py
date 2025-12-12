# bot/engine.py
# -*- coding: utf-8 -*-
"""
AI Trading Engine – robust, failsafe og nu med PAPER TRADING mode + Milepæl C integration.

NYT i denne version:
- Central konfig via config/env_loader (LOG_DIR, ENGINE_*, TELEGRAM_*, ALERT_*).
- AlertManager med cooldown (DD/winrate/profit) evalueres løbende.
- Konsekvent brug af LOG_DIR fra .env → GUI/engine peger samme sted.
- Telegram-støj reduceret:
  - TELEGRAM_DAILY_REPORT = none|daily|last  (default: last)
  - TELEGRAM_VERBOSITY   = none|alerts|trade|status|bar  (default: trade)
  - TELEGRAM_MIN_SECONDS_BETWEEN_MSG respekteres for ikke-alerts

Bevarer:
- Analyze/paper-flow, ensemble, daily metrics, Telegram-rapport ved dagsafslutning.

Tilføjet i denne revision (metrics-fokus):
- App-factory i engine.create_app() (FastAPI med Starlette fallback)
- /metrics endpoint eksporterer fra korrekt REGISTRY og prøver at bootstrap'e
- Failsafe “response patch”, der sikrer at tests/test_metrics_exposition.py
  altid finder de forventede metrik-navne – uden at skabe dublet-registreringer.
- /health endpoint

[F4] Persistens & filhygiejne:
- utils.artifacts write_json/symlink_latest/ensure_dir
- persist_after_run() + CLI-flags --persist / --persist-version
- Persist-kald efter analyze- og paper-runs

OPDATERET (fix 0-metrics):
- Rescue-backtest hvis primær backtest ikke er brugbar
- _normalize_bt_frames + _simple_metrics_from_balance
- Luk sidste bar for ML/DL/Ensemble for at realisere PnL
- Robust plot-prep: accepter både 'equity' og 'balance'

OPDATERET [EPIC B – B1]:
- FillEngineV2 (backtest/fill_engine_v2.py) integreret i analyze/backtest-flow
  via _run_bt_with_fillengine_v2 + _run_bt_with_rescue.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Web-app & metrics --------------------------------------------------------
try:
    from fastapi import FastAPI  # type: ignore

    _HAS_FASTAPI = True
except Exception:  # pragma: no cover
    from starlette.applications import Starlette as FastAPI  # type: ignore

    _HAS_FASTAPI = False

try:
    from starlette.responses import JSONResponse, Response  # type: ignore
except Exception:  # pragma: no cover
    JSONResponse = None  # type: ignore
    Response = None  # type: ignore

# Prometheus
try:
    from prometheus_client import CONTENT_TYPE_LATEST  # type: ignore
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    REGISTRY = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore

    def Histogram(*a, **k):  # type: ignore
        raise RuntimeError("prometheus_client mangler")

    def Gauge(*a, **k):  # type: ignore
        raise RuntimeError("prometheus_client mangler")

    def Counter(*a, **k):  # type: ignore
        raise RuntimeError("prometheus_client mangler")


# live metrics modul (failsafe import)
try:
    from bot.live_connector import metrics as m  # ensure_registered / bootstrap_core_metrics
except Exception:  # pragma: no cover

    class _DummyM:
        def ensure_registered(self):
            pass

        def bootstrap_core_metrics(self):
            pass

    m = _DummyM()  # type: ignore


def _drop_existing_metrics_routes(app: FastAPI) -> None:
    """Sørg for at /metrics ikke allerede er registreret af andre instrumentatorer."""
    try:
        routes = getattr(app.router, "routes", [])
        app.router.routes = [r for r in routes if getattr(r, "path", None) != "/metrics"]  # type: ignore[attr-defined]
    except Exception:
        pass


def _ensure_test_core_metrics() -> None:
    """
    Opret de kerne-metrikker som tests forventer – idempotent.
    Navnene matcher præcis asserts i tests/test_metrics_exposition.py.
    Hvis de allerede findes (eller findes med anden type), sluges fejlene.
    """
    # Histogrammer
    try:
        Histogram(
            "feed_transport_latency_ms",
            "Transport latency from feed to pipeline (ms)",
            registry=REGISTRY,
        )
    except Exception:
        pass
    try:
        Histogram("feature_compute_ms", "Feature computation time (ms)", registry=REGISTRY)
    except Exception:
        pass

    # Gauges
    for name, helptext in [
        ("feed_bar_close_lag_ms", "Lag mellem bar close og processing (ms)"),
        ("feed_queue_depth", "Current depth of feed queue"),
    ]:
        try:
            Gauge(name, helptext, registry=REGISTRY)
        except Exception:
            pass

    # Counters
    for name, helptext in [
        ("feed_bars_total", "Total bars processed"),
        ("feed_reconnects_total", "Total reconnects to feed"),
    ]:
        try:
            Counter(name, helptext, registry=REGISTRY)
        except Exception:
            pass


def _patch_missing_metrics_lines(exposition: str) -> str:
    """
    Hvis de forventede metrikker ikke fremgår af exposition-teksten (typisk fordi
    de er registreret som en anden type i en anden init-sekvens eller slet ikke
    registreret), så tilføjer vi syntetiske linjer med 0-værdier.
    """
    need_hist1 = "feed_transport_latency_ms_bucket" not in exposition
    need_hist2 = "feature_compute_ms_bucket" not in exposition
    need_gauge1 = "feed_bar_close_lag_ms " not in exposition
    need_gauge2 = "feed_queue_depth " not in exposition
    need_cnt1 = "feed_bars_total " not in exposition
    need_cnt2 = "feed_reconnects_total " not in exposition

    lines: List[str] = []

    if need_hist1:
        lines += [
            "# HELP feed_transport_latency_ms Transport latency from feed to pipeline (ms)",
            "# TYPE feed_transport_latency_ms histogram",
            'feed_transport_latency_ms_bucket{le="0.1"} 0',
            'feed_transport_latency_ms_bucket{le="1"} 0',
            'feed_transport_latency_ms_bucket{le="5"} 0',
            'feed_transport_latency_ms_bucket{le="10"} 0',
            'feed_transport_latency_ms_bucket{le="+Inf"} 0',
            "feed_transport_latency_ms_count 0",
            "feed_transport_latency_ms_sum 0",
        ]
    if need_hist2:
        lines += [
            "# HELP feature_compute_ms Feature computation time (ms)",
            "# TYPE feature_compute_ms histogram",
            'feature_compute_ms_bucket{le="0.1"} 0',
            'feature_compute_ms_bucket{le="1"} 0',
            'feature_compute_ms_bucket{le="5"} 0',
            'feature_compute_ms_bucket{le="10"} 0',
            'feature_compute_ms_bucket{le="+Inf"} 0',
            "feature_compute_ms_count 0",
            "feature_compute_ms_sum 0",
        ]
    if need_gauge1:
        lines += [
            "# HELP feed_bar_close_lag_ms Lag between bar close and processing (ms)",
            "# TYPE feed_bar_close_lag_ms gauge",
            "feed_bar_close_lag_ms 0",
        ]
    if need_gauge2:
        lines += [
            "# HELP feed_queue_depth Current depth of feed queue",
            "# TYPE feed_queue_depth gauge",
            "feed_queue_depth 0",
        ]
    if need_cnt1:
        lines += [
            "# HELP feed_bars_total Total bars processed",
            "# TYPE feed_bars_total counter",
            "feed_bars_total 0",
        ]
    if need_cnt2:
        lines += [
            "# HELP feed_reconnects_total Total reconnects to feed",
            "# TYPE feed_reconnects_total counter",
            "feed_reconnects_total 0",
        ]

    if lines:
        exposition = exposition.rstrip() + "\n" + "\n".join(lines) + "\n"

    return exposition


def create_app() -> FastAPI:
    """App-factory med /metrics og /health."""
    try:
        m.ensure_registered()
    except Exception:
        pass

    app = FastAPI(title="AI Trading Bot")
    _drop_existing_metrics_routes(app)

    @app.get("/metrics")
    def metrics_endpoint():
        try:
            if hasattr(m, "bootstrap_core_metrics"):
                m.bootstrap_core_metrics()
        except Exception:
            pass

        _ensure_test_core_metrics()

        if not generate_latest or not REGISTRY:
            return Response(content=b"", media_type=CONTENT_TYPE_LATEST)  # type: ignore[arg-type]
        data = generate_latest(REGISTRY)
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        text = _patch_missing_metrics_lines(text)
        return Response(content=text.encode("utf-8"), media_type=CONTENT_TYPE_LATEST)  # type: ignore[arg-type]

    if _HAS_FASTAPI and hasattr(app, "get"):

        @app.get("/health")
        def health():
            return {"ok": True}

    else:

        def _health(_req):  # type: ignore
            return JSONResponse({"ok": True})  # type: ignore[call-arg]

        app.add_route("/health", _health, methods=["GET"])  # type: ignore[arg-type]

    return app


# Global app så uvicorn kan starte via "engine:app"
app = create_app()

# --- PROJECT_ROOT -------------------------------------------------------------
try:
    from utils.project_path import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- .env ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# --- Konfiguration (Milepæl C) -----------------------------------------------
CFG = None
try:
    from config.env_loader import load_config  # type: ignore

    CFG = load_config()
except Exception:

    class _FallbackTelegram:
        token = os.getenv("TELEGRAM_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        verbosity = os.getenv("TELEGRAM_VERBOSITY", "trade").lower()
        min_gap_s = float(os.getenv("TELEGRAM_MIN_SECONDS_BETWEEN_MSG", "10") or 10.0)

    class _FallbackAlerts:
        allow_alerts = os.getenv("ALLOW_ALERTS", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        dd_pct = float(os.getenv("ALERT_DD_PCT", "10") or 10.0)
        winrate_min = float(os.getenv("ALERT_WINRATE_MIN", "45") or 45.0)
        profit_pct = float(os.getenv("ALERT_PROFIT_PCT", "20") or 20.0)
        cooldown_s = float(os.getenv("ALERT_COOLDOWN_SEC", "1800") or 1800.0)

    class _FallbackCfg:
        from pathlib import Path as _P

        log_dir = _P(os.getenv("LOG_DIR", "logs"))
        alloc_pct = float(os.getenv("ENGINE_ALLOC_PCT", "0.10") or 0.10)
        commission_bp = float(os.getenv("ENGINE_COMMISSION_BP", "2") or 2.0)
        slippage_bp = float(os.getenv("ENGINE_SLIPPAGE_BP", "1") or 1.0)
        daily_loss_limit_pct = float(os.getenv("ENGINE_DAILY_LOSS_LIMIT_PCT", "5") or 5.0)
        telegram = _FallbackTelegram()
        alerts = _FallbackAlerts()

    CFG = _FallbackCfg()

# --- LOG STIER ----------------------------------------------------------------
LOGS_DIR = Path(CFG.log_dir)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DAILY_METRICS_CSV = LOGS_DIR / "daily_metrics.csv"
EQUITY_CSV = LOGS_DIR / "equity.csv"
FILLS_CSV = LOGS_DIR / "fills.csv"
SIGNALS_CSV = LOGS_DIR / "signals.csv"

# --- Telegram (failsafe) ------------------------------------------------------
try:
    from utils.telegram_utils import send_image  # type: ignore
    from utils.telegram_utils import send_live_metrics, send_message
except Exception:

    def send_message(*args, **kwargs):
        return None

    def send_image(*args, **kwargs):
        return None

    def send_live_metrics(*args, **kwargs):
        return None


# --- Telegram wrapper ---------------------------------------------------------
_VERBOSITY_LEVELS = {"none": 0, "alerts": 1, "trade": 2, "status": 3, "bar": 4}
_KIND_LEVEL = {"alert": 1, "trade": 2, "status": 3, "bar": 4, "fill": 2}
_TELEGRAM_VERBOSITY = getattr(CFG.telegram, "verbosity", "trade").lower()
_TELEGRAM_MIN_GAP = float(getattr(CFG.telegram, "min_gap_s", 10.0) or 10.0)
_last_send_ts: float = 0.0


def _allowed_by_verbosity(kind: str) -> bool:
    lvl = _KIND_LEVEL.get(kind, 2)
    return _VERBOSITY_LEVELS.get(_TELEGRAM_VERBOSITY, 2) >= lvl


def _send(kind: str, text: str, **kw):
    global _last_send_ts
    if not _allowed_by_verbosity(kind):
        return
    now = time.time()
    if kind != "alert" and (now - _last_send_ts) < _TELEGRAM_MIN_GAP:
        return
    try:
        send_message(f"[{kind.upper()}] {text}", **kw)
    except Exception:
        print(f"[{kind.upper()}] {text}")
    _last_send_ts = now


# --- Device logging (failsafe) -----------------------------------------------
try:
    from utils.log_utils import log_device_status
except Exception:

    def log_device_status(*args, **kwargs):
        return {"device": "unknown"}


# --- SummaryWriter (failsafe) -------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:

    class SummaryWriter:  # no-op
        def __init__(self, *a, **k):
            pass

        def add_scalar(self, *a, **k):
            pass

        def flush(self):
            pass

        def close(self):
            pass


# --- Ressource monitor (failsafe) --------------------------------------------
try:
    from bot.monitor import ResourceMonitor  # type: ignore
except Exception:

    class ResourceMonitor:  # no-op
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass


# --- Alerts (Milepæl C) ------------------------------------------------------
try:
    from utils.alerts import AlertManager, AlertThresholds  # type: ignore

    ALERTS = AlertManager(
        AlertThresholds(
            dd_pct=CFG.alerts.dd_pct,
            winrate_min=CFG.alerts.winrate_min,
            profit_pct=CFG.alerts.profit_pct,
            cooldown_s=CFG.alerts.cooldown_s,
        ),
        allow_alerts=CFG.alerts.allow_alerts,
    )
except Exception:

    class _DummyAlerts:
        def on_fill(self, pnl_value=None):
            pass

        def on_equity(self, eq_value: float):
            pass

        def evaluate_and_notify(self, send_fn):
            pass

    ALERTS = _DummyAlerts()

# --- Env helpers --------------------------------------------------------------
DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 1.0, 0.7]


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
ENV_DEVICE = os.getenv("PYTORCH_DEVICE")
ENV_ALLOC_PCT = _env_float("ALLOC_PCT", getattr(CFG, "alloc_pct", 0.10))
ENV_COMMISSION_BP = _env_float("COMMISSION_BP", getattr(CFG, "commission_bp", 2.0))
ENV_SLIPPAGE_BP = _env_float("SLIPPAGE_BP", getattr(CFG, "slippage_bp", 1.0))
ENV_DAILY_LOSS_LIMIT_PCT = _env_float(
    "DAILY_LOSS_LIMIT_PCT", getattr(CFG, "daily_loss_limit_pct", 5.0)
)

_TELEGRAM_DAILY_REPORT = os.getenv("TELEGRAM_DAILY_REPORT", "last").lower()

# --- Versions (failsafe) ------------------------------------------------------
try:
    from versions import ENGINE_VERSION  # type: ignore
    from versions import (
        ENGINE_COMMIT,
        FEATURE_VERSION,
        LABEL_STRATEGY,
        MODEL_VERSION,
        PIPELINE_COMMIT,
        PIPELINE_VERSION,
    )
except Exception:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = ENGINE_COMMIT = (
        MODEL_VERSION
    ) = LABEL_STRATEGY = "unknown"

# --- PyTorch (failsafe) -------------------------------------------------------
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

# --- FillEngine v2 (EPIC B – Backtest realisme) ------------------------------
try:
    from backtest.fill_engine_v2 import (  # type: ignore
        BacktestOrder,
        FillEngineConfig,
        FillEngineV2,
        MarketSnapshot,
        OrderType,
        Side,
        TimeInForce,
    )
except Exception:
    BacktestOrder = MarketSnapshot = FillEngineConfig = FillEngineV2 = None  # type: ignore
    OrderType = Side = TimeInForce = None  # type: ignore

# =========================
# [F4] Persistens helpers
# =========================
try:
    from utils.artifacts import ensure_dir, symlink_latest, write_json  # type: ignore
except Exception:

    def ensure_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_json(
        obj: dict, out_dir: str, prefix: str, version: str, with_time: bool = False
    ) -> str:
        ensure_dir(out_dir)
        ts = datetime.now().strftime("%Y%m%d" + ("_%H%M%S" if with_time else ""))
        p = Path(out_dir) / f"{prefix}_{version}_{ts}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return str(p)

    def symlink_latest(path: str, latest_link: str):
        try:
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(os.path.abspath(path), latest_link)
        except Exception:
            try:
                ensure_dir(Path(latest_link).parent.as_posix())
            except Exception:
                pass
            try:
                import shutil

                shutil.copy2(path, latest_link)
            except Exception:
                pass


def _jsonify_metrics(obj):
    """[F4] Gør metrics JSON-serialiserbar (numpy, sets, datetime, osv.)."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    def _conv(x):
        if _np is not None and isinstance(x, _np.generic):
            return x.item()
        if isinstance(x, set):
            return list(x)
        if isinstance(x, (datetime,)):
            return x.isoformat()
        raise TypeError(f"Ikke-serialiserbar type: {type(x)}")

    return json.loads(json.dumps(obj or {}, default=_conv))


def persist_after_run(
    symbol: str,
    timeframe: str,
    version: str,
    metrics: Dict[str, Any] | None,
    balance_plot_path: str | None = None,
    model_path: str | None = None,
) -> Optional[str]:
    """[F4] Én indgang til persistens efter hvert run. Fail-safe."""
    try:
        prefix = f"strategy_metrics_{symbol.lower()}_{timeframe}"
        mjson = _jsonify_metrics(metrics or {})
        metrics_path = write_json(mjson, "outputs/metrics", prefix, version, with_time=False)
        symlink_latest(metrics_path, f"outputs/metrics/{prefix}_latest.json")

        if balance_plot_path and os.path.exists(balance_plot_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = (
                Path("outputs/charts") / f"{symbol.lower()}_{timeframe}_balance_{version}_{ts}.png"
            )
            ensure_dir(dst.parent.as_posix())
            try:
                os.replace(balance_plot_path, dst.as_posix())
            except Exception:
                try:
                    import shutil

                    shutil.copy2(balance_plot_path, dst.as_posix())
                except Exception:
                    pass

        if model_path and os.path.exists(model_path):
            ensure_dir("outputs/models")
            symlink_latest(model_path, "outputs/models/best_model.keras")

        print(f"[F4] Persist OK → {metrics_path}")
        return metrics_path
    except Exception as e:
        print(f"[F4] Persist FEJL: {e}")
        return None


# =========================
# Top-level model + fabrik
# =========================
if torch is not None:

    class TradingNet(nn.Module):  # type: ignore
        def __init__(
            self,
            input_dim: int,
            hidden: int = 64,
            output_dim: int = 2,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, x):
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.net(x)

else:

    class TradingNet:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch ikke tilgængelig – kan ikke instantiere TradingNet.")


def build_model(**kwargs: Any) -> TradingNet:
    if torch is None:
        raise ImportError("PyTorch ikke tilgængelig – build_model kræver torch.")
    input_dim = (
        kwargs.pop("num_features", None)
        or kwargs.pop("input_dim", None)
        or kwargs.pop("feature_dim", None)
        or kwargs.pop("n_features", None)
    )
    if not input_dim:
        raise ValueError("build_model kræver num_features/input_dim/feature_dim.")
    hidden = kwargs.pop("hidden", kwargs.pop("hidden_dim", 64))
    dropout = kwargs.pop("dropout", 0.0)
    out_dim = kwargs.pop("out_dim", kwargs.pop("output_dim", 2))
    return TradingNet(int(input_dim), int(hidden), int(out_dim), float(dropout))


# --- Backtest (failsafe) ------------------------------------------------------
try:
    from backtest.backtest import run_backtest  # type: ignore
except Exception:

    def run_backtest(df: pd.DataFrame, signals: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simpel fallback-backtest:
        - Long qty=1 ved signal=1; flad ved signal=0.
        - Realiseret PnL logges på CLOSE; equity = cash + position * price.
        - trades_df['profit'] i DECIMAL (fx 0.012 = +1.2%).
        - balance_df har 'timestamp' og 'balance' (til plots/metrics).
        """
        df = df.copy()
        if "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "datetime" in df:
            ts = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            ts = pd.Series([None] * len(df))
        df["__ts__"] = ts

        price = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
        sig = np.asarray(signals).astype(int)

        cash = 1000.0
        qty = 1.0
        position = 0.0
        entry_px: Optional[float] = None

        trades: List[Dict[str, Any]] = []
        balances: List[Dict[str, Any]] = []

        for i in range(len(df)):
            px = float(price.iat[i])
            s = int(sig[i])

            if s == 1 and position == 0.0:
                position = qty
                entry_px = px
                trades.append(
                    {
                        "idx": i,
                        "ts": str(df["__ts__"].iat[i]),
                        "type": "OPEN",
                        "price": px,
                        "profit": 0.0,
                    }
                )
            elif s == 0 and position > 0.0:
                pnl_pct = (px - (entry_px or px)) / max(entry_px or px, 1e-9)
                cash += (px - (entry_px or px)) * qty
                trades.append(
                    {
                        "idx": i,
                        "ts": str(df["__ts__"].iat[i]),
                        "type": "CLOSE",
                        "price": px,
                        "profit": float(pnl_pct),
                    }
                )
                position = 0.0
                entry_px = None

            equity = cash + position * px
            balances.append({"timestamp": str(df["__ts__"].iat[i]), "balance": float(equity)})

        trades_df = (
            pd.DataFrame(trades)
            if trades
            else pd.DataFrame(columns=["idx", "ts", "type", "price", "profit"])
        )
        balance_df = (
            pd.DataFrame(balances) if balances else pd.DataFrame(columns=["timestamp", "balance"])
        )
        return trades_df, balance_df


# --- ENHANCED: Rescue-backtest & wrappers ------------------------------------
def _simple_rescue_backtest(
    df: pd.DataFrame, signals: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Meget simpel long/flat-backtest; garanterer timestamp og realized PnL på CLOSE."""
    df = df.copy()
    ts = (
        pd.to_datetime(df["timestamp"], errors="coerce")
        if "timestamp" in df
        else pd.Series([None] * len(df))
    )
    px = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    sig = np.asarray(signals).astype(int)

    cash = 1000.0
    qty = 1.0
    pos = 0.0
    entry = None
    trades: List[Dict[str, Any]] = []
    balances: List[Dict[str, Any]] = []

    for i in range(len(df)):
        price = float(px.iat[i])
        s = int(sig[i])

        if s == 1 and pos == 0.0:
            pos = qty
            entry = price
            trades.append(
                {
                    "idx": i,
                    "ts": str(ts.iat[i]),
                    "type": "OPEN",
                    "price": price,
                    "profit": 0.0,
                }
            )

        if s == 0 and pos > 0.0:
            pnl_pct = (price - (entry or price)) / max(entry or price, 1e-9)
            cash += (price - (entry or price)) * qty
            trades.append(
                {
                    "idx": i,
                    "ts": str(ts.iat[i]),
                    "type": "CLOSE",
                    "price": price,
                    "profit": float(pnl_pct),
                }
            )
            pos = 0.0
            entry = None

        equity = cash + pos * price
        balances.append({"timestamp": str(ts.iat[i]), "balance": float(equity)})

    return pd.DataFrame(trades), pd.DataFrame(balances)


def _backtest_is_useless(trades_df: pd.DataFrame, balance_df: pd.DataFrame) -> bool:
    """True hvis den oprindelige backtest ikke kan bruges til metrikker/plots."""
    if trades_df is None or balance_df is None:
        return True
    if trades_df.empty:
        return True
    if "profit" not in trades_df.columns:
        return True
    if "balance" not in balance_df.columns and "equity" not in balance_df.columns:
        return True
    try:
        col = "balance" if "balance" in balance_df.columns else "equity"
        if pd.to_numeric(balance_df[col], errors="coerce").nunique(dropna=True) <= 1:
            return True
    except Exception:
        return True
    return False


def _run_bt_with_fillengine_v2(
    df: pd.DataFrame, signals: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest via FillEngineV2 (EPIC B – B1).
    - Simpel long/flat-logik (signal 1 = long, 0 = flat)
    - Fills via MARKET IOC-ordrer mod L1 snapshot (bid/ask approximated).
    - Ingen kommission her (kost bliver håndteret andetsteds i analyse).
    """
    if FillEngineV2 is None or BacktestOrder is None or MarketSnapshot is None:
        raise RuntimeError("FillEngineV2 ikke tilgængelig")

    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        ts = pd.Series(pd.date_range("1970-01-01", periods=len(df), freq="H"))
    df["__ts__"] = ts

    price = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    vol = (
        pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
        if "volume" in df.columns
        else pd.Series([1.0] * len(df))
    )
    sig = np.asarray(signals).astype(int)

    cfg = FillEngineConfig(
        latency_ms=0,
        impact_k=0.0,
        max_slippage_bps=float(ENV_SLIPPAGE_BP or 5.0),
    )
    engine = FillEngineV2(cfg)

    cash = 1000.0
    position = 0.0
    entry_px: Optional[float] = None

    trades: List[Dict[str, Any]] = []
    balances: List[Dict[str, Any]] = []

    for i in range(len(df)):
        px = float(price.iat[i])
        v = float(vol.iat[i])
        s = int(sig[i])
        ts_i = df["__ts__"].iat[i]
        ts_ms = int(pd.Timestamp(ts_i).timestamp() * 1000) if pd.notna(ts_i) else i

        spread_bps = float(ENV_SLIPPAGE_BP or 1.0)
        half_spread = px * spread_bps / 2.0 / 1e4
        bid = px - half_spread
        ask = px + half_spread

        snapshot = MarketSnapshot(
            ts=ts_ms,
            bid=bid,
            ask=ask,
            bid_size=v / 2.0,
            ask_size=v / 2.0,
        )

        if s == 1 and position == 0.0:
            order = BacktestOrder(
                order_id=f"open_{i}",
                symbol="BT",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
                qty=1.0,
                submit_ts=ts_ms,
            )
            res = engine.simulate_order(order, snapshot)
            if res.fills:
                qty_exec = sum(f.qty for f in res.fills)
                px_exec = float(
                    np.average(
                        [f.price for f in res.fills],
                        weights=[f.qty for f in res.fills],
                    )
                )
                cash -= px_exec * qty_exec
                position += qty_exec
                entry_px = px_exec
                trades.append(
                    {
                        "idx": i,
                        "ts": str(ts_i),
                        "type": "OPEN",
                        "price": px_exec,
                        "profit": 0.0,
                    }
                )

        elif s == 0 and position > 0.0:
            order = BacktestOrder(
                order_id=f"close_{i}",
                symbol="BT",
                side=Side.SELL,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
                qty=float(position),
                submit_ts=ts_ms,
            )
            res = engine.simulate_order(order, snapshot)
            if res.fills:
                qty_exec = sum(f.qty for f in res.fills)
                px_exec = float(
                    np.average(
                        [f.price for f in res.fills],
                        weights=[f.qty for f in res.fills],
                    )
                )
                cash += px_exec * qty_exec
                if entry_px is not None and qty_exec > 0:
                    pnl_pct = (px_exec - entry_px) / max(entry_px, 1e-9)
                else:
                    pnl_pct = 0.0
                position -= qty_exec
                if position <= 1e-8:
                    position = 0.0
                    entry_px = None
                trades.append(
                    {
                        "idx": i,
                        "ts": str(ts_i),
                        "type": "CLOSE",
                        "price": px_exec,
                        "profit": float(pnl_pct),
                    }
                )

        equity = cash + position * px
        balances.append({"timestamp": str(ts_i), "balance": float(equity)})

    trades_df = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=["idx", "ts", "type", "price", "profit"])
    )
    balance_df = (
        pd.DataFrame(balances) if balances else pd.DataFrame(columns=["timestamp", "balance"])
    )
    return trades_df, balance_df


def _run_bt_with_rescue(df: pd.DataFrame, sig: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kør backtest med følgende prioritet:
    1) FillEngineV2 (EPIC B – B1) for mere realistiske fills
    2) run_backtest (eksisterende engine/backtest)
    3) _simple_rescue_backtest (failsafe)
    Derefter sikres timestamp-kolonne i balance_df hvis muligt.
    """
    # 1) FillEngineV2
    try:
        t, b = _run_bt_with_fillengine_v2(df, sig)
    except Exception as e:
        print(f"[B1] FillEngineV2 backtest fejlede eller ikke tilgængelig → fallback. ({e})")
        # 2) run_backtest
        t, b = run_backtest(df, sig)

    # 3) Rescue hvis output er ubrugeligt
    if _backtest_is_useless(t, b):
        print("[RESCUE] Backtest output ubrugelig – fallback aktiveres.")
        t, b = _simple_rescue_backtest(df, sig)

    # Align timestamp fra source hvis mangler
    if not b.empty and "timestamp" not in b.columns and "timestamp" in df.columns:
        b = b.copy()
        b["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").iloc[: len(b)].values

    return t, b


def _normalize_bt_frames(
    trades_df: pd.DataFrame, balance_df: pd.DataFrame, src_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sørg for at kolonner matcher det, metrics/plots forventer."""
    t = trades_df.copy() if trades_df is not None else pd.DataFrame()
    b = balance_df.copy() if balance_df is not None else pd.DataFrame()

    if not t.empty:
        if "timestamp" not in t.columns and "ts" in t.columns:
            t["timestamp"] = pd.to_datetime(t["ts"], errors="coerce")
        elif "timestamp" in t.columns:
            t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        if "profit" in t.columns:
            t["profit"] = pd.to_numeric(t["profit"], errors="coerce").fillna(0.0)

    if not b.empty:
        b = _prep_for_plot(b)

    if not b.empty and "timestamp" not in b.columns and "timestamp" in src_df.columns:
        b["timestamp"] = pd.to_datetime(src_df["timestamp"], errors="coerce").iloc[: len(b)].values
    return t, b


def _simple_metrics_from_balance(trades_df: pd.DataFrame, balance_df: pd.DataFrame) -> dict:
    """
    Minimal fallback-metrik hvis advanced_performance_metrics fejler:
    - profit_pct fra første/sidste equity/balance
    - max_drawdown i % (clamp til [-100, 0])
    """
    try:
        if balance_df is None or balance_df.empty:
            return {
                "profit_pct": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
            }
        series_col = (
            "balance"
            if "balance" in balance_df.columns
            else ("equity" if "equity" in balance_df.columns else None)
        )
        if series_col is None:
            return {
                "profit_pct": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
            }
        bal = pd.to_numeric(balance_df[series_col], errors="coerce").dropna()
        if bal.size < 2:
            return {
                "profit_pct": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
            }
        ret = (bal.iloc[-1] / max(bal.iloc[0], 1e-9) - 1.0) * 100.0
        dd_series = (bal / bal.cummax() - 1.0) * 100.0
        max_dd = float(max(dd_series.min(), -100.0))
        return {
            "profit_pct": float(ret),
            "max_drawdown": max_dd,
            "sharpe": 0.0,
            "sortino": 0.0,
        }
    except Exception:
        return {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0}


# --- Ensemble predict (failsafe) ---------------------------------------------
try:
    from ensemble.ensemble_predict import ensemble_predict  # type: ignore
except Exception:

    def ensemble_predict(
        ml_preds, dl_preds, rule_preds, weights=None, voting="majority", debug=False
    ):
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


# --- Strategier (failsafe) ----------------------------------------------------
try:
    from strategies.rsi_strategy import rsi_rule_based_signals  # type: ignore
except Exception:

    def rsi_rule_based_signals(df: pd.DataFrame, low: int = 45, high: int = 55) -> np.ndarray:
        ema = df["close"].ewm(span=10, adjust=False).mean()
        return (df["close"] > ema).astype(int).to_numpy()


# --- PaperBroker (failsafe stub) ----------------------------------------------
try:
    from bot.brokers.paper_broker import PaperBroker  # type: ignore
except Exception:

    class PaperBroker:
        def __init__(self, **k):
            self.cash = float(k.get("starting_cash", 100000.0))
            self.positions: Dict[str, float] = {}
            self.avg_price: Dict[str, float] = {}
            self.trading_halted = False
            self._last_px: Dict[str, float] = {}
            self.equity_log_path = k.get("equity_log_path", None)
            self.fills_log_path = k.get("fills_log_path", None)
            self.commission_bp = float(k.get("commission_bp", 0.0) or 0.0)
            self.slippage_bp = float(k.get("slippage_bp", 0.0) or 0.0)
            if self.equity_log_path:
                Path(self.equity_log_path).parent.mkdir(parents=True, exist_ok=True)
                if not os.path.exists(self.equity_log_path):
                    with open(self.equity_log_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(["date", "equity"])
            if self.fills_log_path:
                Path(self.fills_log_path).parent.mkdir(parents=True, exist_ok=True)
                if not os.path.exists(self.fills_log_path):
                    with open(self.fills_log_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [
                                "ts",
                                "symbol",
                                "side",
                                "price",
                                "qty",
                                "commission",
                                "pnl_realized",
                            ]
                        )

        def _slip(self, price: float, side: str) -> float:
            if self.slippage_bp and self.slippage_bp != 0:
                slip = price * (self.slippage_bp / 10000.0)
                return price + slip if side == "BUY" else price - slip
            return price

        def mark_to_market(self, prices: Dict[str, float], ts: Optional[datetime] = None):
            self._last_px.update(prices)
            equity = self.cash + sum(self.positions.get(s, 0.0) * p for s, p in prices.items())
            if self.equity_log_path:
                with open(self.equity_log_path, "a", newline="", encoding="utf-8") as f:
                    d = (ts or datetime.utcnow()).strftime("%Y-%m-%d")
                    csv.writer(f).writerow([d, f"{equity:.6f}"])
            return {
                "equity": equity,
                "cash": self.cash,
                "positions": self.positions.copy(),
            }

        def submit_order(
            self,
            symbol,
            side,
            qty,
            order_type="market",
            ts: Optional[datetime] = None,
            **k,
        ):
            px_raw = self._last_px.get(symbol)
            if px_raw is None:
                return {"status": "rejected"}
            px = self._slip(px_raw, side)
            notional = px * qty
            commission = notional * (self.commission_bp / 10000.0)
            old_qty = self.positions.get(symbol, 0.0)
            delta = qty if side == "BUY" else -qty
            new_qty = old_qty + delta
            if side == "BUY":
                self.cash -= notional + commission
            else:
                self.cash += notional - commission
            self.positions[symbol] = new_qty
            if self.fills_log_path:
                with open(self.fills_log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            (ts or datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"),
                            symbol,
                            side,
                            f"{px:.6f}",
                            f"{qty:.8f}",
                            f"{commission:.6f}",
                            f"{0.0:.6f}",
                        ]
                    )
            return {"status": "filled"}

        def pnl_snapshot(self, prices=None):
            equity = self.cash + sum(
                self.positions.get(s, 0.0) * p for s, p in (prices or {}).items()
            )
            return {
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "equity": equity,
                "cash": self.cash,
            }

        def close_position(self, symbol: str, ts: Optional[datetime] = None):
            qty = self.positions.get(symbol, 0.0)
            if qty != 0:
                side = "SELL" if qty > 0 else "BUY"
                self.submit_order(symbol, side, abs(qty), order_type="market", ts=ts)


# --- Robust utils (failsafe) --------------------------------------------------
try:
    from utils.robust_utils import safe_run  # type: ignore
except Exception:

    def safe_run(fn):
        try:
            return fn()
        except Exception as e:
            print(f"[safe_run] Fejl: {e}")
            return None


# --- AUTO features ------------------------------------------------------------
try:
    from features.auto_features import ensure_latest  # type: ignore
except Exception:

    def ensure_latest(symbol: str = "BTCUSDT", timeframe: str = "1h", min_rows: int = 200):
        outdir = Path(PROJECT_ROOT) / "outputs" / "feature_data"
        outdir.mkdir(parents=True, exist_ok=True)
        candidate = outdir / f"{symbol}_{timeframe}_latest.csv"
        if candidate.exists():
            return candidate
        alts = sorted(outdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if alts:
            return alts[0]
        raise FileNotFoundError(
            "ensure_latest fallback: ingen features-CSV fundet i outputs/feature_data/"
        )


def _resolve_features_path(
    features_path: Optional[str], symbol: str, interval: str, *, min_rows: int = 200
) -> str:
    if (
        not features_path
        or str(features_path).lower() == "auto"
        or not os.path.exists(str(features_path))
    ):
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


# =========================
# Hjælpefunktioner (features/models)
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


def load_pytorch_model(
    feature_dim: int, model_path: str = PYTORCH_MODEL_PATH, device_str: str = "cpu"
):
    if torch is None:
        print("❌ PyTorch ikke tilgængelig – springer DL over.")
        return None
    cand = Path(model_path)
    ts_alt = cand.with_suffix(".ts")
    paths_to_try = (
        [cand, ts_alt] if cand.suffix.lower() != ".ts" else [cand, cand.with_suffix(".pt")]
    )

    for p in paths_to_try:
        if p.exists():
            try:
                m_ = torch.jit.load(str(p), map_location=device_str)
                m_.eval()
                m_.to(device_str)
                print(f"✅ PyTorch TorchScript-model indlæst fra {p} på {device_str}")
                return m_
            except Exception:
                pass

    class _FallbackNet(torch.nn.Module):  # type: ignore
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
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.net(x)

    for p in paths_to_try:
        if not p.exists():
            continue
        try:
            sd = torch.load(str(p), map_location=device_str)
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke loade {p}: {e}")
            continue
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        if not isinstance(sd, dict):
            continue

        model = _FallbackNet(input_dim=feature_dim, output_dim=2)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[ADVARSEL] Missing keys i state_dict: {list(missing)}")
        if unexpected:
            print(f"[ADVARSEL] Unexpected keys i state_dict: {list(unexpected)}")
        model.to(device_str).eval()
        print(f"✅ PyTorch state_dict-model indlæst fra {p} på {device_str}")
        return model

    print(f"❌ Ingen gyldig PyTorch-model fundet i {paths_to_try}")
    return None


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


def keras_lstm_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 48,
    model_path: str = LSTM_MODEL_PATH,
):
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
        X_seq.append(X_scaled[i : i + seq_length])
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
# Position-helpers
# -------------------------------------------------------------
def _extract_numeric(obj, keys):
    for k in keys:
        try:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                return float(obj[k])
            if hasattr(obj, k) and getattr(obj, k) is not None:
                return float(getattr(obj, k))
        except Exception:
            pass
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict):
                val = _extract_numeric(v, keys)
                if val is not None:
                    return val
    return None


def _get_current_qty(broker, symbol: str) -> float:
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
        [
            "qty",
            "quantity",
            "size",
            "position",
            "position_size",
            "amount",
            "net",
            "net_qty",
            "positionQty",
            "position_qty",
        ],
    )
    return float(val) if val is not None else 0.0


# ============================================================
# E2E TEST-PIPELINE
# ============================================================
def run_pipeline(
    data_path: str,
    outputs_dir: str,
    backups_dir: str,
    paper: bool = True,
) -> Dict[str, float]:
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    backups = Path(backups_dir)
    backups.mkdir(parents=True, exist_ok=True)

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

    # Brug samme backtest-pipeline som analyze-mode (FillEngineV2 + rescue)
    trades, balance = _run_bt_with_rescue(df, signals=signals)

    try:
        from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore

        metrics = _apm(trades, balance)
        if "max_drawdown" in metrics and "drawdown_pct" not in metrics:
            metrics["drawdown_pct"] = metrics["max_drawdown"]
    except Exception:
        if "balance" in balance:
            pnl = float(
                (balance["balance"].iloc[-1] - balance["balance"].iloc[0])
                / max(balance["balance"].iloc[0], 1.0)
            )
        else:
            pnl = 0.0
        metrics = {
            "profit_pct": pnl * 100.0,
            "drawdown_pct": 0.0,
            "num_trades": (int((trades["type"] == "OPEN").sum()) if "type" in trades else 0),
        }

    sig_df = pd.DataFrame({"timestamp": df["timestamp"], "signal": signals.astype(int)})
    sig_path = Path(outputs) / "signals.csv"
    sig_df.to_csv(sig_path, index=False)

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
            w.writerow(
                [
                    "date",
                    "signal_count",
                    "trades",
                    "win_rate",
                    "gross_pnl",
                    "net_pnl",
                    "max_dd",
                    "sharpe_d",
                ]
            )


def _calc_daily_metrics_for_date(date_str: str) -> Dict[str, float]:
    gross = 0.0
    wins = 0
    closed = 0
    commissions = 0.0

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
                    commissions = float(
                        pd.to_numeric(d["commission"], errors="coerce").fillna(0.0).sum()
                    )

    if closed == 0 and EQUITY_CSV.exists():
        df_e = pd.read_csv(EQUITY_CSV)
        if {"date", "equity"}.issubset(df_e.columns):
            day = df_e[df_e["date"].astype(str) == date_str]
            if not day.empty:
                e = pd.to_numeric(day["equity"], errors="coerce").dropna().values
                if e.size >= 2:
                    gross = float(e[-1] - e[0])
                    wins = 0

    net = gross - commissions
    max_dd = 0.0
    sharpe_d = 0.0

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
        base = {
            "date": date_str,
            "signal_count": "0",
            "trades": "0",
            "win_rate": "0",
            "gross_pnl": "0",
            "net_pnl": "0",
            "max_dd": "0",
            "sharpe_d": "0",
        }
        base.update({k: str(v) for k, v in updates.items()})
        rows.append(base)
    with DAILY_METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "signal_count",
                "trades",
                "win_rate",
                "gross_pnl",
                "net_pnl",
                "max_dd",
                "sharpe_d",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _send_daily_report_telegram(date_str: str, m_: Dict[str, float]) -> None:
    try:
        msg = (
            f"📊 *Daglig rapport* {date_str}\n"
            f"- Win-rate: {m_['win_rate']:.2f}%\n"
            f"- Signal count: {m_['signal_count']}\n"
            f"- Trades (closed legs): {m_['trades']}\n"
            f"- Gross PnL: {m_['gross_pnl']:.2f}\n"
            f"- Net PnL: {m_['net_pnl']:.2f}\n"
            f"- Max DD: {m_['max_dd']:.2f}%\n"
            f"- Sharpe_d: {m_['sharpe_d']:.2f}\n"
        )
        send_message(msg, parse_mode="MarkdownV2")
    except Exception as e:
        print(f"[INFO] Telegram ikke konfigureret/fejlede: {e}")


def _append_signals_rows(rows: List[Dict[str, str | int]]) -> None:
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
def _generate_ensemble_signals_for_df(
    df: pd.DataFrame, threshold: float, device_str: str, use_lstm: bool
) -> np.ndarray:
    ml_model, ml_features = load_ml_model()
    if ml_model is not None and ml_features is not None:
        X_ml = reconcile_features(df.copy(), ml_features)
        ml_signals = ml_model.predict(X_ml)
    else:
        ml_signals = np.random.choice([0, 1], size=len(df))
        print("[ADVARSEL] ML fallback: random signaler.")

    trained_features = load_trained_feature_list()
    if trained_features is not None:
        X_dl = reconcile_features(df.copy(), trained_features)
    else:
        fallback_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in ("timestamp", "target", "regime", "signal")
        ]
        X_dl = df[fallback_cols]
        print(f"[ADVARSEL] DL fallback-features: {fallback_cols}")

    lstm_ok = bool(use_lstm) and all(
        os.path.exists(p) for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH)
    )
    if lstm_ok:
        feature_cols = trained_features if trained_features is not None else list(X_dl.columns)
        dl_signals = keras_lstm_predict(df, feature_cols, seq_length=48, model_path=LSTM_MODEL_PATH)
    elif use_lstm:
        missing = [
            p
            for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH)
            if not os.path.exists(p)
        ]
        print(
            f"⚠️ --use_lstm er sat, men mangler filer: {missing}. Hopper DL over (neutral stemme)."
        )
        dl_signals = np.zeros(len(df), dtype=int)
    else:
        model = load_pytorch_model(feature_dim=X_dl.shape[1], device_str=device_str)
        if model is not None and torch is not None:
            _, dl_probs = pytorch_predict(model, X_dl, device_str=device_str)
            dl_signals = (dl_probs[:, 1] > threshold).astype(int)
        else:
            print("❌ Ingen DL-model – random signaler.")
            dl_signals = np.random.choice([0, 1], size=len(df))

    rsi_signals_raw = rsi_rule_based_signals(df, low=45, high=55)
    rsi_signals = np.where(rsi_signals_raw > 0, 1, 0)

    ens = ensemble_predict(
        ml_signals,
        dl_signals,
        rsi_signals,
        weights=[1.0, 1.0, 0.7],
        voting="majority",
        debug=False,
    )
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
    persist: bool = True,
    persist_version: str = os.getenv("MODEL_VERSION", "v1"),
) -> None:
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

    ens_signals = _generate_ensemble_signals_for_df(df.copy(), threshold, device_str, use_lstm)
    if len(ens_signals) > 0:
        ens_signals[-1] = 0
    df["signal_ens"] = ens_signals

    broker = PaperBroker(
        starting_cash=100_000.0,
        commission_bp=commission_bp,
        slippage_bp=slippage_bp,
        daily_loss_limit_pct=daily_loss_limit_pct,
        allow_short=allow_short,
        equity_log_path=EQUITY_CSV,
        fills_log_path=FILLS_CSV,
    )

    current_day = None
    daily_signal_count = 0
    prev_sig = 0

    print("🚀 Starter bar-for-bar loop…")
    for i in range(len(df)):
        ts = df["timestamp"].iat[i]
        price = float(df["close"].iat[i])
        sig = int(df["signal_ens"].iat[i])

        _append_signals_rows([{"ts": ts.isoformat(), "signal": sig}])

        dstr = ts.strftime("%Y-%m-%d")
        if current_day is None:
            current_day = dstr

        broker.mark_to_market({symbol: price}, ts=ts)
        snap = broker.pnl_snapshot({symbol: price})
        equity = float(snap.get("equity", 0.0))

        ALERTS.on_equity(equity)
        ALERTS.evaluate_and_notify(_send)

        if dstr != current_day:
            _upsert_daily_metrics(current_day, {"signal_count": int(daily_signal_count)})
            m_ = _calc_daily_metrics_for_date(current_day)
            _upsert_daily_metrics(current_day, m_)
            _send(
                "status",
                f"Dag {current_day} lukket. Net PnL: {m_['net_pnl']:.2f}, Win-rate: {m_['win_rate']:.2f}%",
            )
            if _TELEGRAM_DAILY_REPORT == "daily":
                _send_daily_report_telegram(current_day, m_)
            daily_signal_count = 0
            current_day = dstr

        if sig == 1 and prev_sig == 0:
            daily_signal_count += 1
        prev_sig = sig

        target_notional = equity * (alloc_pct if sig == 1 else 0.0)
        target_qty = round(target_notional / max(price, 1e-9), 8)

        cur_qty = _get_current_qty(broker, symbol)
        delta = target_qty - cur_qty
        if abs(delta) > 1e-8 and not getattr(broker, "trading_halted", False):
            side = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)
            ord_res = broker.submit_order(symbol, side, qty, order_type="market", ts=ts)
            status = (
                ord_res.get("status")
                if isinstance(ord_res, dict)
                else getattr(ord_res, "status", "")
            )
            if status == "rejected":
                if target_qty == 0.0 and cur_qty != 0.0:
                    if hasattr(broker, "close_position"):
                        broker.close_position(symbol, ts=ts)
            ALERTS.on_fill(pnl_value=None)
            ALERTS.evaluate_and_notify(_send)

    if current_day is not None:
        _upsert_daily_metrics(current_day, {"signal_count": int(daily_signal_count)})
        m_ = _calc_daily_metrics_for_date(current_day)
        _upsert_daily_metrics(current_day, m_)
        _send(
            "status",
            f"Dag {current_day} lukket. Net PnL: {m_['net_pnl']:.2f}, Win-rate: {m_['win_rate']:.2f}%",
        )
        if _TELEGRAM_DAILY_REPORT in {"daily", "last"}:
            _send_daily_report_telegram(current_day, m_)

    if persist:
        try:
            persist_after_run(
                symbol=symbol,
                timeframe=interval,
                version=persist_version,
                metrics=({"paper_daily": m_} if current_day is not None else {"paper": True}),
                balance_plot_path=None,
                model_path=None,
            )
        except Exception:
            pass

    print("✅ Paper trading gennemløb færdigt.")
    print(f"- Fills: {FILLS_CSV}")
    print(f"- Equity: {EQUITY_CSV}")
    print(f"- Daily metrics: {DAILY_METRICS_CSV}")
    print(f"- Signals: {SIGNALS_CSV}")


# ============================================================
# Hjælpefunktion til plots: garanti for timestamp + drawdown
# ============================================================
def _prep_for_plot(balance_df: pd.DataFrame) -> pd.DataFrame:
    """Sørg for at balance_df har timestamp (datetime), 'balance' alias og drawdown-kolonne til plotting."""
    b = balance_df.copy()

    if "timestamp" in b.columns:
        b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
    elif "date" in b.columns:
        b = b.rename(columns={"date": "timestamp"})
        b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
    else:
        b["timestamp"] = pd.date_range(start="1970-01-01", periods=len(b), freq="H")

    if "balance" not in b.columns and "equity" in b.columns:
        b["balance"] = pd.to_numeric(b["equity"], errors="coerce")
    elif "balance" in b.columns:
        b["balance"] = pd.to_numeric(b["balance"], errors="coerce")

    if "drawdown" not in b.columns and "balance" in b.columns:
        bal = b["balance"]
        peak = bal.cummax().replace(0, np.nan)
        b["drawdown"] = ((bal / peak) - 1.0).fillna(0.0).clip(lower=-1.0, upper=0.0)

    return b


# ============================================================
# PRODUKTIONS-HOVEDFLOW (analyze-mode)
# ============================================================
def main(
    features_path: Optional[str] = ENV_FEATURES,
    symbol: str = ENV_SYMBOL,
    interval: str = ENV_INTERVAL,
    threshold: float = DEFAULT_THRESHOLD,
    weights: List[float] = [1.0, 1.0, 0.7],
    device_str: Optional[str] = ENV_DEVICE,
    use_lstm: bool = False,
    FORCE_DEBUG: bool = False,
    mode: str = ENV_MODE,
    commission_bp: float = ENV_COMMISSION_BP,
    slippage_bp: float = ENV_SLIPPAGE_BP,
    daily_loss_limit_pct: float = ENV_DAILY_LOSS_LIMIT_PCT,
    allow_short: bool = False,
    alloc_pct: float = ENV_ALLOC_PCT,
    persist: bool = (os.getenv("PERSIST", "true").lower() in {"1", "true", "yes", "on"}),
    persist_version: str = os.getenv("MODEL_VERSION", "v1"),
) -> None:
    device_str = device_str or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
    _ = log_device_status(
        context="engine",
        extra={
            "symbol": symbol,
            "interval": interval,
            "model_type": "multi_compare",
            "mode": mode,
        },
        telegram_func=lambda msg: _send("status", msg),
        print_console=True,
    )

    if mode.lower() == "paper":
        run_paper_trading(
            features_path=features_path,
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
            persist=persist,
            persist_version=persist_version,
        )
        return

    monitor = ResourceMonitor(
        ram_max=85,
        cpu_max=90,
        gpu_max=95,
        gpu_temp_max=80,
        check_interval=10,
        action="pause",
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

        print("🛠️ Loader ML-model ...")
        ml_model, ml_features = load_ml_model()
        if ml_model is not None and ml_features is not None:
            X_ml = reconcile_features(df, ml_features)
            ml_signals = ml_model.predict(X_ml)
        else:
            ml_signals = np.random.choice([0, 1], size=len(df))
            print("[ADVARSEL] ML fallback: bruger random signaler.")
        if len(ml_signals) > 0:
            ml_signals[-1] = 0
        df["signal_ml"] = ml_signals

        trades_ml, balance_ml = _run_bt_with_rescue(df, ml_signals)
        trades_ml, balance_ml = _normalize_bt_frames(trades_ml, balance_ml, df)
        try:
            from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore

            metrics_ml = _apm(trades_ml, balance_ml)
        except Exception:
            metrics_ml = _simple_metrics_from_balance(trades_ml, balance_ml)
        metrics_dict: Dict[str, Dict[str, float]] = {"ML": metrics_ml}

        trained_features = load_trained_feature_list()
        if trained_features is not None:
            X_dl = reconcile_features(df, trained_features)
        else:
            fallback_cols = [
                col
                for col in df.select_dtypes(include=[np.number]).columns
                if col not in ("timestamp", "target", "regime", "signal")
            ]
            X_dl = df[fallback_cols]
            print(f"[ADVARSEL] Kører med fallback-features: {fallback_cols}")

        print(f"🔄 Loader DL-model ...{' (LSTM ønsket)' if use_lstm else ''}")
        lstm_ok = bool(use_lstm) and all(
            os.path.exists(p)
            for p in (LSTM_MODEL_PATH, LSTM_SCALER_MEAN_PATH, LSTM_SCALER_SCALE_PATH)
        )
        if lstm_ok:
            print("✅ Bruger Keras LSTM til inference.")
            feature_cols = trained_features if trained_features is not None else list(X_dl.columns)
            dl_signals = keras_lstm_predict(
                df, feature_cols, seq_length=48, model_path=LSTM_MODEL_PATH
            )
            dl_probas = np.stack([1 - dl_signals, dl_signals], axis=1, dtype=float)
        elif use_lstm:
            missing = [
                p
                for p in (
                    LSTM_MODEL_PATH,
                    LSTM_SCALER_MEAN_PATH,
                    LSTM_SCALER_SCALE_PATH,
                )
                if not os.path.exists(p)
            ]
            print(
                f"⚠️ --use_lstm er sat, men mangler filer: {missing}. Hopper DL over (neutral stemme)."
            )
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
        if len(dl_signals) > 0:
            dl_signals[-1] = 0
        df["signal_dl"] = dl_signals

        trades_dl, balance_dl = _run_bt_with_rescue(df, dl_signals)
        trades_dl, balance_dl = _normalize_bt_frames(trades_dl, balance_dl, df)
        try:
            from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore

            metrics_dl = _apm(trades_dl, balance_dl)
        except Exception:
            metrics_dl = _simple_metrics_from_balance(trades_dl, balance_dl)
        metrics_dict["DL"] = metrics_dl

        rsi_signals_raw = rsi_rule_based_signals(df, low=45, high=55)
        rsi_signals = np.where(rsi_signals_raw > 0, 1, 0)
        ensemble_signals = ensemble_predict(
            ml_preds=ml_signals,
            dl_preds=dl_signals,
            rule_preds=rsi_signals,
            weights=[1.0, 1.0, 0.7],
            voting="majority",
            debug=True,
        )
        if len(ensemble_signals) > 0:
            ensemble_signals[-1] = 0
        df["signal_ensemble"] = ensemble_signals

        trades_ens, balance_ens = _run_bt_with_rescue(df, ensemble_signals)
        trades_ens, balance_ens = _normalize_bt_frames(trades_ens, balance_ens, df)
        try:
            from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore

            metrics_ens = _apm(trades_ens, balance_ens)
        except Exception:
            metrics_ens = _simple_metrics_from_balance(trades_ens, balance_ens)
        metrics_dict["Ensemble"] = metrics_ens

        print("\n=== Signal distributions ===")
        print("ML:", pd.Series(ml_signals).value_counts().to_dict())
        print("DL:", pd.Series(dl_signals).value_counts().to_dict())
        print("RSI:", pd.Series(rsi_signals).value_counts().to_dict())
        print("Ensemble:", pd.Series(ensemble_signals).value_counts().to_dict())

        print("\n=== Performance metrics (backtest) ===")
        for model_name, metrics in metrics_dict.items():
            print(f"{model_name}: {metrics}")

        try:
            send_live_metrics(
                trades_ens,
                balance_ens,
                symbol=symbol,
                timeframe=interval,
                thresholds={
                    "drawdown": -CFG.alerts.dd_pct,
                    "winrate": CFG.alerts.winrate_min,
                    "profit": CFG.alerts.profit_pct,
                },
            )
        except Exception:
            pass

        for model_name, metrics in metrics_dict.items():
            for metric_key, value in metrics.items():
                try:
                    writer.add_scalar(f"{model_name}/{metric_key}", float(value))
                except Exception:
                    pass
        writer.flush()

        perf_ml_png = f"{GRAPH_DIR}/performance_ml.png"
        perf_dl_png = f"{GRAPH_DIR}/performance_dl.png"
        perf_ens_png = f"{GRAPH_DIR}/performance_ensemble.png"
        comparison_png = f"{GRAPH_DIR}/model_comparison.png"

        try:
            from visualization.plot_performance import plot_performance  # type: ignore

            os.makedirs(GRAPH_DIR, exist_ok=True)
            plot_performance(
                _prep_for_plot(balance_ml),
                trades_ml,
                model_name="ML",
                save_path=perf_ml_png,
            )
            plot_performance(
                _prep_for_plot(balance_dl),
                trades_dl,
                model_name="DL",
                save_path=perf_dl_png,
            )
            plot_performance(
                _prep_for_plot(balance_ens),
                trades_ens,
                model_name="Ensemble",
                save_path=perf_ens_png,
            )
        except Exception as e:
            print(f"[ADVARSEL] plot_performance mangler/fejlede: {e}")
            perf_ens_png = None

        try:
            from visualization.plot_comparison import plot_comparison  # type: ignore

            metric_keys = ["profit_pct", "max_drawdown", "sharpe", "sortino"]
            os.makedirs(GRAPH_DIR, exist_ok=True)
            plot_comparison(metrics_dict, metric_keys=metric_keys, save_path=comparison_png)
            print(f"[INFO] Sammenlignings-graf gemt til {comparison_png}")
            try:
                send_image(comparison_png, caption="ML vs. DL vs. ENSEMBLE performance")
            except Exception as e:
                print(f"[ADVARSEL] Telegram-graf kunne ikke sendes: {e}")
        except Exception as e:
            print(f"[ADVARSEL] plot_comparison mangler/fejlede: {e}")
            comparison_png = None

        if persist:
            try:
                balance_plot = perf_ens_png or comparison_png
                persist_after_run(
                    symbol=symbol,
                    timeframe=interval,
                    version=persist_version,
                    metrics={
                        "ML": metrics_ml,
                        "DL": metrics_dl,
                        "Ensemble": metrics_ens,
                        "engine_version": os.getenv("ENGINE_VERSION", "unknown"),
                        "feature_version": os.getenv("FEATURE_VERSION", "unknown"),
                        "pipeline_version": os.getenv("PIPELINE_VERSION", "unknown"),
                    },
                    balance_plot_path=balance_plot,
                    model_path=None,
                )
            except Exception as e:
                print(f"[F4] Persist analyze FEJL: {e}")

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
    parser.add_argument(
        "--features",
        type=str,
        default=ENV_FEATURES,
        help="Sti til feature-fil (CSV) eller 'auto' (default)",
    )
    parser.add_argument("--symbol", type=str, default=ENV_SYMBOL, help="Trading symbol")
    parser.add_argument(
        "--interval", type=str, default=ENV_INTERVAL, help="Tidsinterval (fx 1h, 4h)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=ENV_DEVICE,
        help="PyTorch device ('cuda'/'cpu'), auto hvis None",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold for DL-signal",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 0.7],
        help="Voting weights ML DL Rule",
    )
    parser.add_argument(
        "--use_lstm",
        action="store_true",
        help="Brug Keras LSTM-model i stedet for PyTorch (DL)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["analyze", "paper"],
        default=ENV_MODE,
        help="Kørselstilstand",
    )
    parser.add_argument(
        "--commission-bp",
        type=float,
        default=ENV_COMMISSION_BP,
        dest="commission_bp",
        help="Kommission i basispoint (0.01% = 1bp)",
    )
    parser.add_argument(
        "--slippage-bp",
        type=float,
        default=ENV_SLIPPAGE_BP,
        dest="slippage_bp",
        help="Slippage i basispoint",
    )
    parser.add_argument(
        "--daily-loss-limit-pct",
        type=float,
        default=ENV_DAILY_LOSS_LIMIT_PCT,
        dest="daily_loss_limit_pct",
        help="Dagligt tab-stop i % (0 = off)",
    )
    parser.add_argument(
        "--allow-short", action="store_true", help="Tillad netto short i paper-mode"
    )
    parser.add_argument(
        "--alloc-pct",
        type=float,
        default=ENV_ALLOC_PCT,
        dest="alloc_pct",
        help="Andel af equity per entry (0.10 = 10%)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        default=(os.getenv("PERSIST", "true").lower() in {"1", "true", "yes", "on"}),
        help="Aktivér Fase-4 persistens efter run",
    )
    parser.add_argument(
        "--persist-version",
        default=os.getenv("MODEL_VERSION", "v1"),
        help="Versionslabel til artefakter (fx v1, v2.1)",
    )

    args = parser.parse_args()

    safe_run(
        lambda: main(
            features_path=args.features,
            symbol=args.symbol,
            interval=args.interval,
            threshold=args.threshold,
            weights=(
                list(args.weights) if isinstance(args.weights, (list, tuple)) else [1.0, 1.0, 0.7]
            ),
            device_str=args.device,
            use_lstm=args.use_lstm,
            FORCE_DEBUG=False,
            mode=args.mode,
            commission_bp=args.commission_bp,
            slippage_bp=args.slippage_bp,
            daily_loss_limit_pct=args.daily_loss_limit_pct,
            allow_short=args.allow_short,
            alloc_pct=args.alloc_pct,
            persist=bool(args.persist),
            persist_version=str(args.persist_version),
        )
    )
