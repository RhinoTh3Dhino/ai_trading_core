# gui/app.py
"""
AI Trading GUI – Live Monitor + Backtest

Denne version er hærdet ift.:
- Robust API-kald (konfigurerbart timeout via GUI_HTTP_TIMEOUT, default 8s)
- Tolerant CSV-fallback (engine='python', on_bad_lines='skip')
- Ensrettet normalisering af live-signaler
- Små UX-forbedringer (Clear cache, bedre fejltekster)
"""
from __future__ import annotations

import os
import sys
import io
import json
import argparse
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

# Plot-backend uden display
os.environ.setdefault("MPLBACKEND", "Agg")

# --- bootstrap PYTHONPATH så imports virker fra 'gui/' ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# requests er valgfrit (GUI falder tilbage til lokale filer hvis ikke installeret)
try:
    import requests
    _HAS_REQ = True
except Exception:
    requests = None
    _HAS_REQ = False

# Auto-refresh (valgfrit)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*_, **__):
        return 0

# Projekt-imports
from utils.project_path import PROJECT_ROOT
from backtest.backtest import run_backtest
from utils.metrics_utils import advanced_performance_metrics
from features.features_pipeline import generate_features, load_features

# Strategi-impls (valgfrit – falder tilbage til interne)
try:
    from strategies.rsi_strategy import rsi_rule_based_signals as _rsi_signals_lib  # type: ignore
except Exception:
    _rsi_signals_lib = None

try:
    from strategies.ema_cross_strategy import ema_cross_signals as _ema_signals_lib  # type: ignore
except Exception:
    _ema_signals_lib = None

try:
    from strategies.macd_strategy import macd_cross_signals as _macd_signals_lib  # type: ignore
except Exception:
    _macd_signals_lib = None

# Streamlit
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None
    _HAS_ST = False


# -------------------------
# Hjælpefunktioner
# -------------------------
ROOT = Path(PROJECT_ROOT)
LOGS = ROOT / "logs"
API_DIR = ROOT / "api"

GUI_HTTP_TIMEOUT = float(os.getenv("GUI_HTTP_TIMEOUT", "8"))  # sek.

def _robust_read_csv(path: Path) -> pd.DataFrame:
    """Tålmodig CSV-reader – skipper skæve linjer og ignorerer encodingfejl."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        try:
            # sidste udvej: læs tekst, filtrér linjer der ligner CSV
            from io import StringIO
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for ln in f:
                    if ("," in ln) or ln.lower().startswith("date") or ln.lower().startswith("ts"):
                        lines.append(ln)
            return pd.read_csv(StringIO("".join(lines)), engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Sørg for en 'timestamp' kolonne og sorter."""
    out = df.copy()
    if "timestamp" not in out.columns:
        for cand in ("datetime", "Timestamp"):
            if cand in out.columns:
                out = out.rename(columns={cand: "timestamp"})
                break
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    gain = (d.where(d > 0, 0.0)).rolling(length).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Sikrer ema_200 og atr_14 findes til filtre (overskriver ikke hvis de allerede er der)."""
    out = df.copy()
    if "close" not in out.columns:
        return out
    c = out["close"].astype(float)
    # EMA200
    if "ema_200" not in out.columns and "ema200" not in out.columns:
        out["ema_200"] = _ema(c, 200)
    elif "ema200" in out.columns and "ema_200" not in out.columns:
        out = out.rename(columns={"ema200": "ema_200"})
    # ATR14
    if "atr_14" not in out.columns and {"high", "low", "close"}.issubset(out.columns):
        out["atr_14"] = _atr(out["high"].astype(float), out["low"].astype(float), c, 14)
    return out

def _apply_filters(
    raw_sig: np.ndarray,
    df: pd.DataFrame,
    *,
    position_mode: str = "both",
    regime: str = "none",
    cooldown: int = 0,
    min_atr_pct: float = 0.0,
) -> np.ndarray:
    sig = raw_sig.astype(int).copy()
    n = len(sig)
    if n == 0:
        return sig

    if position_mode == "long_only":
        sig = (sig > 0).astype(int)
    elif position_mode == "short_only":
        sig = np.zeros_like(sig, dtype=int)

    if regime in ("price_vs_ema200", "ema200_slope"):
        df_i = _prepare_indicators(df)
        c = df_i["close"].astype(float)
        ema200 = df_i["ema_200"].astype(float)
        if regime == "price_vs_ema200":
            long_ok = c > ema200
            short_ok = c < ema200
        else:
            slope = ema200.diff()
            long_ok = slope > 0
            short_ok = slope < 0
        for i in range(n):
            if sig[i] == 1 and not bool(long_ok.iloc[i]):
                sig[i] = 0
            if i > 0 and raw_sig[i] == 0 and raw_sig[i-1] == 1 and not bool(short_ok.iloc[i]):
                sig[i] = 1

    if min_atr_pct and "atr_14" in df.columns:
        atr_pct = (df["atr_14"].astype(float) / df["close"].astype(float)).fillna(0.0) * 100.0
        for i in range(1, n):
            flipped = (sig[i] != sig[i-1])
            if flipped and atr_pct.iloc[i] < float(min_atr_pct):
                sig[i] = sig[i-1]

    if cooldown and cooldown > 0:
        last_flip = 0
        for i in range(1, n):
            if sig[i] != sig[i-1]:
                if (i - last_flip) <= cooldown:
                    sig[i] = sig[i-1]
                else:
                    last_flip = i
    return sig

def _signals_rsi(
    df: pd.DataFrame,
    *,
    length: int = 14,
    low: int = 45,
    high: int = 55,
    position_mode: str = "both",
    regime: str = "none",
    cooldown: int = 0,
    min_atr_pct: float = 0.0,
) -> np.ndarray:
    if _rsi_signals_lib is not None:
        try:
            base = np.asarray(_rsi_signals_lib(df, low=low, high=high)).astype(int)
            return _apply_filters(base, _prepare_indicators(df),
                                  position_mode=position_mode, regime=regime,
                                  cooldown=cooldown, min_atr_pct=min_atr_pct)
        except Exception:
            pass
    rs = _calc_rsi(df["close"].astype(float), length).fillna(50)
    raw = np.where(rs > high, 0, np.where(rs < low, 1, 0)).astype(int)
    return _apply_filters(raw, _prepare_indicators(df),
                          position_mode=position_mode, regime=regime,
                          cooldown=cooldown, min_atr_pct=min_atr_pct)

def _signals_ema(
    df: pd.DataFrame,
    *,
    fast: int = 9,
    slow: int = 21,
    position_mode: str = "both",
    regime: str = "none",
    cooldown: int = 0,
    min_atr_pct: float = 0.0,
) -> np.ndarray:
    if _ema_signals_lib is not None:
        try:
            base = np.asarray(_ema_signals_lib(df, short=fast, long=slow)).astype(int)
            return _apply_filters(base, _prepare_indicators(df),
                                  position_mode=position_mode, regime=regime,
                                  cooldown=cooldown, min_atr_pct=min_atr_pct)
        except Exception:
            pass
    c = df["close"].astype(float)
    ema_f = _ema(c, fast)
    ema_s = _ema(c, slow)
    raw = (ema_f > ema_s).astype(int).to_numpy()
    return _apply_filters(raw, _prepare_indicators(df),
                          position_mode=position_mode, regime=regime,
                          cooldown=cooldown, min_atr_pct=min_atr_pct)

def _signals_macd(
    df: pd.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    position_mode: str = "both",
    regime: str = "none",
    cooldown: int = 0,
    min_atr_pct: float = 0.0,
) -> np.ndarray:
    if _macd_signals_lib is not None:
        try:
            base = np.asarray(_macd_signals_lib(df)).astype(int)
            return _apply_filters(base, _prepare_indicators(df),
                                  position_mode=position_mode, regime=regime,
                                  cooldown=cooldown, min_atr_pct=min_atr_pct)
        except Exception:
            pass
    c = df["close"].astype(float)
    ema_f = _ema(c, fast)
    ema_s = _ema(c, slow)
    macd = ema_f - ema_s
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    raw = (macd > macd_sig).astype(int).to_numpy()
    return _apply_filters(raw, _prepare_indicators(df),
                          position_mode=position_mode, regime=regime,
                          cooldown=cooldown, min_atr_pct=min_atr_pct)

def _signals_ensemble(
    df: pd.DataFrame,
    *,
    rsi_len: int = 14, rsi_low: int = 45, rsi_high: int = 55,
    ema_fast: int = 9, ema_slow: int = 21,
    macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
    position_mode: str = "both",
    regime: str = "none",
    cooldown: int = 0,
    min_atr_pct: float = 0.0,
) -> np.ndarray:
    a = _signals_rsi(df, length=rsi_len, low=rsi_low, high=rsi_high,
                     position_mode="both", regime="none", cooldown=0, min_atr_pct=0.0)
    b = _signals_ema(df, fast=ema_fast, slow=ema_slow,
                     position_mode="both", regime="none", cooldown=0, min_atr_pct=0.0)
    c = _signals_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal,
                      position_mode="both", regime="none", cooldown=0, min_atr_pct=0.0)
    mat = np.vstack([a, b, c]).astype(int)
    raw = (mat.sum(axis=0) >= 2).astype(int)
    return _apply_filters(raw, _prepare_indicators(df),
                          position_mode=position_mode, regime=regime,
                          cooldown=cooldown, min_atr_pct=min_atr_pct)

def _silent_read(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)

def _silent_run_backtest(df: pd.DataFrame, signals: np.ndarray):
    return _silent_read(run_backtest, df, signals=signals)

def _backtest_and_metrics(df: pd.DataFrame, signals: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    trades, balance = _silent_run_backtest(df, signals=signals)
    metrics = advanced_performance_metrics(trades, balance)
    return trades, balance, metrics

def _plot_equity(balance: pd.DataFrame, title: str = "Equity curve") -> plt.Figure:
    fig = plt.figure(figsize=(8, 3))
    ax = fig.gca()
    x = pd.to_datetime(balance.get("timestamp", pd.RangeIndex(len(balance))))
    if "equity" in balance:
        y = balance["equity"]
    elif "balance" in balance:
        y = balance["balance"]
    else:
        y = balance.select_dtypes(include=[np.number]).iloc[:, -1]
    ax.plot(x, y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

def _save_artifacts(trades: pd.DataFrame, balance: pd.DataFrame, metrics: Dict, tag: str) -> Path:
    outdir = Path(PROJECT_ROOT) / "outputs" / "gui" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(outdir / "trades.csv", index=False)
    balance.to_csv(outdir / "balance.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return outdir


# -------------------------
# Live Monitor helpers (API + fallback) – cache
# -------------------------
if _HAS_ST:
    @st.cache_data(ttl=2.0, show_spinner=False)
    def _http_get_json(url: str):
        if not _HAS_REQ:
            return None, "requests ikke installeret"
        try:
            r = requests.get(url, timeout=GUI_HTTP_TIMEOUT, headers={"accept": "application/json"})
            if not r.ok:
                return None, f"HTTP {r.status_code}"
            # prøv JSON først – ellers tekst→JSON
            try:
                return r.json(), None
            except Exception:
                try:
                    return json.loads(r.text), None
                except Exception:
                    return None, "Kunne ikke parse JSON-svar."
        except Exception as e:
            return None, str(e)

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_status(api_base: str) -> Tuple[Dict, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/status")
        if data is not None:
            return data, None
        return {"last_run_ts": datetime.utcnow().isoformat(), "mode": "paper", "win_rate_7d": 0.0, "drawdown_pct": 0.0}, err

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_equity(api_base: str) -> Tuple[pd.DataFrame, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/equity")
        if isinstance(data, list):
            try:
                return pd.DataFrame(data), None
            except Exception:
                pass
        p = LOGS / "equity.csv"
        if p.exists():
            return _robust_read_csv(p)[["date", "equity"]], f"API fejl: {err}. Viser CSV."
        return pd.DataFrame(columns=["date", "equity"]), "Ingen equity-data."

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_metrics(api_base: str, limit: int = 30) -> Tuple[pd.DataFrame, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/metrics/daily?limit={limit}")
        if isinstance(data, list):
            try:
                return pd.DataFrame(data), None
            except Exception:
                pass
        p = LOGS / "daily_metrics.csv"
        if p.exists():
            return _robust_read_csv(p).tail(limit), f"API fejl: {err}. Viser CSV."
        return pd.DataFrame(), "Ingen metrikker."

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_signals(api_base: str) -> Tuple[List[Dict], Optional[str]]:
        data, err = _http_get_json(f"{api_base}/signals/latest")
        if data:
            if isinstance(data, dict):
                return [data], None
            if isinstance(data, list):
                return data[-20:], None
        p = API_DIR / "sim_signals.json"
        if p.exists():
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(arr, list):
                    return arr[-20:], "API utilgængelig — viser lokale mock-signaler."
            except Exception as e:
                return [], f"Kan ikke læse mock-signaler: {e}"
        return [], "Ingen signaler."

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_fills(api_base: str, limit: int = 50) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Hent seneste handler.
        1) Prøver API: /fills?limit=N (hvis tilgængeligt).
        2) Fallback: logs/fills.csv.
        Normaliserer kolonner til: ts, symbol, side, qty, price, commission, pnl_realized.
        """
        data, err = _http_get_json(f"{api_base}/fills?limit={limit}")
        if isinstance(data, list) and data:
            try:
                df = pd.DataFrame(data)
                return df, None
            except Exception:
                pass

        # Fallback CSV
        p = LOGS / "fills.csv"
        if not p.exists():
            return pd.DataFrame(), "Ingen fills fundet endnu."
        try:
            df = _robust_read_csv(p)
            lower = {c: c.lower() for c in df.columns}
            df = df.rename(columns=lower)
            rename_map = {
                "timestamp": "ts",
                "time": "ts",
                "datetime": "ts",
                "quantity": "qty",
                "pnl": "pnl_realized",
            }
            for src, dst in rename_map.items():
                if src in df.columns and dst not in df.columns:
                    df = df.rename(columns={src: dst})
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            key = "ts" if "ts" in df.columns else None
            if key:
                df = df.sort_values(key, ascending=False)
            df = df.head(limit).reset_index(drop=True)
            cols = [c for c in ["ts", "symbol", "side", "qty", "price", "pnl_realized", "commission"] if c in df.columns]
            return (df[cols] if cols else df), None
        except Exception as e:
            return pd.DataFrame(), f"Kunne ikke læse fills.csv: {e}"

    # --- AI helpers ---
    @st.cache_data(ttl=1.0, show_spinner=False)
    def lm_ai_explain(api_base: str, i: int = 0, context_bars: int = 60) -> Tuple[str, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/ai/explain_trade?i={i}&context_bars={context_bars}")
        if isinstance(data, dict) and "text" in data:
            return str(data["text"]), None
        return "", err or "Intet AI-svar."

    @st.cache_data(ttl=1.0, show_spinner=False)
    def lm_ai_summary(api_base: str, limit_days: int = 7) -> Tuple[str, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/ai/summary?limit_days={limit_days}")
        if isinstance(data, dict) and "text" in data:
            return str(data["text"]), None
        return "", err or "Intet AI-svar."


# -------------------------
# Streamlit helpers (cache) – backtest
# -------------------------
if _HAS_ST:
    @st.cache_data(show_spinner=False)
    def _cached_latest(symbol: str, timeframe: str) -> pd.DataFrame:
        return _silent_read(load_features, symbol, timeframe, version_prefix=None)


# -------------------------
# Streamlit GUI
# -------------------------
def _normalize_signals_df(sigs: List[Dict]) -> pd.DataFrame:
    """Normaliserer signaler til DataFrame med: when, symbol, side, confidence, price, regime."""
    if not sigs:
        return pd.DataFrame(columns=["when", "symbol", "side", "confidence", "price", "regime"])
    df = pd.DataFrame(sigs).copy()

    # parse tidspunkt
    when = None
    if "ts" in df.columns:
        with contextlib.suppress(Exception):
            when = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(None)
    if when is None and "timestamp" in df.columns:
        try:
            s = pd.to_numeric(df["timestamp"], errors="coerce")
            unit = "ms" if s.dropna().max() and float(s.dropna().max()) > 1e11 else "s"
            when = pd.to_datetime(s, errors="coerce", unit=unit, utc=True).dt.tz_convert(None)
        except Exception:
            when = pd.Series([pd.NaT] * len(df))
    if when is None:
        when = pd.Series([pd.NaT] * len(df))
    df["when"] = when

    # numeric normalisering
    for col in ("confidence", "price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_cols = ["when"] + [c for c in ["symbol", "side", "confidence", "price", "regime"] if c in df.columns]
    out = df[out_cols].copy()
    out = out.sort_values("when", ascending=False, na_position="last").reset_index(drop=True)
    return out

def _render_live_monitor():
    st.subheader("📡 Live Monitor (Paper Trading)")
    api_default = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    api_base = st.text_input("API base URL", value=st.session_state.get("api_base", api_default))
    st.session_state["api_base"] = api_base
    interval = st.slider("Auto-refresh (sek.)", 1, 10, 3, 1)
    st_autorefresh(interval=interval * 1000, key="live_refresh_v2")

    # cache-kontrol
    c0, _ = st.columns([1, 5])
    with c0:
        if st.button("🔄 Clear cache"):
            st.cache_data.clear()
            st.toast("Cache ryddet", icon="🧹")

    # Status
    status, status_err = lm_load_status(api_base)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Seneste kørsel (UTC)", str(status.get("last_run_ts", "")).replace("T", " ").split(".")[0])
    with c2:
        st.metric("Mode", status.get("mode", "paper"))
    with c3:
        st.metric("Win-rate (7d)", f'{float(status.get("win_rate_7d", 0.0)):.2f}%')
    with c4:
        st.metric("Drawdown", f'{float(status.get("drawdown_pct", 0.0)):.2f}%')
    if status_err:
        st.info(f"Status fra fallback: {status_err}")

    # Live signaler
    st.markdown("### Live signaler")
    sigs, sig_err = lm_load_signals(api_base)
    if sig_err:
        st.info(sig_err)
    if sigs:
        df_sig = _normalize_signals_df(sigs)
        if not df_sig.empty:
            st.dataframe(df_sig, width='stretch')
        else:
            st.warning("Ingen gyldige signaler at vise.")
    else:
        st.caption("Ingen signaler endnu.")

    # Fills (seneste handler)
    st.markdown("### Fills (seneste handler)")
    df_fills, fills_err = lm_load_fills(api_base, limit=50)
    if fills_err:
        st.info(fills_err)
    if not df_fills.empty:
        st.dataframe(df_fills, width='stretch')
    else:
        st.caption("Ingen handler registreret endnu.")

    # AI-værktøjer
    st.markdown("### 🧠 AI-værktøjer")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Forklar seneste handel (Claude)", use_container_width=False):
            text, err = lm_ai_explain(api_base, i=0, context_bars=60)
            if err:
                st.error(err)
            st.text_area("Forklaring", value=text or "(tomt)", height=220)
    with a2:
        if st.button("AI-summary (7 dage)", use_container_width=False):
            text, err = lm_ai_summary(api_base, limit_days=7)
            if err:
                st.error(err)
            st.text_area("AI-summary", value=text or "(tomt)", height=220)

    # Equity
    st.markdown("### Equity-kurve")
    df_eq, eq_err = lm_load_equity(api_base)
    if eq_err:
        st.info(eq_err)
    if not df_eq.empty:
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(df_eq["date"], errors="coerce"), pd.to_numeric(df_eq["equity"], errors="coerce"))
        ax.set_xlabel("Dato"); ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    else:
        st.caption("Ingen equity-data endnu.")

    # Dagsmetrikker
    st.markdown("### Daglige metrikker (seneste 30 dage)")
    df_m, m_err = lm_load_metrics(api_base, 30)
    if m_err:
        st.info(m_err)
    if not df_m.empty:
        st.dataframe(df_m, width='stretch')
    else:
        st.caption("Ingen metrikker endnu.")

def _render_backtest():
    st.subheader("🧪 Backtest")
    st.session_state.setdefault("df_features", None)
    st.session_state.setdefault("last_results", None)

    with st.form("controls"):
        st.markdown("#### ⚙️ Inputs")
        mode = st.radio("Datakilde", ["Indlæs features (anbefalet)", "Generér fra rå OHLCV"], horizontal=True)
        cols = st.columns(3)
        with cols[0]:
            symbol = st.text_input("Symbol", "BTCUSDT")
        with cols[1]:
            timeframe = st.text_input("Timeframe", "1h")
        with cols[2]:
            strategy = st.selectbox("Strategi", ["RSI", "EMA Cross", "MACD", "Ensemble"])

        st.markdown("#### 🎛️ Strategi-parametre")
        if strategy == "RSI":
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_len = st.slider("RSI længde", 5, 50, 14, 1)
            with col2:
                rsi_low = st.slider("RSI low (contrarian købsbånd)", 0, 60, 45, 1)
            with col3:
                rsi_high = st.slider("RSI high (contrarian salgsbånd)", 40, 100, 55, 1)
        elif strategy == "EMA Cross":
            col1, col2 = st.columns(2)
            with col1:
                ema_fast = st.slider("EMA (hurtig)", 2, 60, 9, 1)
            with col2:
                ema_slow = st.slider("EMA (langsom)", 5, 200, 21, 1)
        elif strategy == "MACD":
            col1, col2, col3 = st.columns(3)
            with col1:
                macd_fast = st.slider("MACD fast", 2, 48, 12, 1)
            with col2:
                macd_slow = st.slider("MACD slow", 5, 96, 26, 1)
            with col3:
                macd_signal = st.slider("MACD signal", 2, 30, 9, 1)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_len = st.slider("RSI længde", 5, 50, 14, 1)
                ema_fast = st.slider("EMA (hurtig)", 2, 60, 9, 1)
            with col2:
                rsi_low = st.slider("RSI low", 0, 60, 45, 1)
                macd_fast = st.slider("MACD fast", 2, 48, 12, 1)
            with col3:
                rsi_high = st.slider("RSI high", 40, 100, 55, 1)
                macd_slow = st.slider("MACD slow", 5, 96, 26, 1)
            macd_signal = st.slider("MACD signal", 2, 30, 9, 1)

        st.markdown("#### 🧠 Globale filtre")
        colf1, colf2, colf3, colf4 = st.columns(4)
        with colf1:
            position_mode = st.selectbox("Positionstype", ["Long & Short", "Kun Long", "Kun Short"])
        with colf2:
            regime = st.selectbox("Regime-filter", ["Ingen", "Pris vs EMA200", "EMA200-slope"])
        with colf3:
            cooldown = st.slider("Vent N bar efter flip (debounce)", 0, 10, 2, 1)
        with colf4:
            min_atr_pct = st.slider("Min ATR% ved entry (0 = slukket)", 0.0, 5.0, 0.0, 0.1,
                                    help="ATR14 / Close * 100.")

        run_from_form = st.form_submit_button("Kør backtest")

    position_map = {"Long & Short": "both", "Kun Long": "long_only", "Kun Short": "short_only"}
    regime_map = {"Ingen": "none", "Pris vs EMA200": "price_vs_ema200", "EMA200-slope": "ema200_slope"}

    df_features: Optional[pd.DataFrame] = None

    if mode == "Indlæs features (anbefalet)":
        uploaded = st.file_uploader(
            "Upload en features-CSV (eller indlæs 'latest' fra outputs/feature_data)",
            type=["csv"],
        )
        if uploaded is not None:
            df_features = pd.read_csv(uploaded, engine="python", on_bad_lines="skip")
            st.session_state.df_features = df_features
            st.success(f"Indlæst upload: {len(df_features)} rækker")
        else:
            use_latest = st.checkbox("Indlæs seneste (latest) fra disk", True)
            if use_latest:
                try:
                    df_features = _cached_latest(symbol, timeframe)
                    st.session_state.df_features = df_features
                    st.info(f"Indlæst seneste features fra disk: {len(df_features)} rækker")
                except Exception as e:
                    st.warning(f"Kunne ikke indlæse seneste features: {e}")
    else:
        st.markdown("**Generér features fra rå OHLCV-CSV**")
        raw = st.file_uploader(
            "Upload rå OHLCV CSV (mindst: timestamp, open, high, low, close, volume)",
            type=["csv"],
        )
        if raw is not None:
            raw_df = pd.read_csv(raw, engine="python", on_bad_lines="skip")
            try:
                df_features = generate_features(
                    raw_df,
                    feature_config=dict(
                        coerce_timestamps=True,
                        patterns_enabled=True,
                        target_mode="direction",
                        horizon=1,
                        dropna=True,
                        normalize=False,
                        drop_all_nan_cols=True,
                    ),
                )
                st.session_state.df_features = df_features
                st.success(f"Features genereret: {df_features.shape}")
            except Exception as e:
                st.error(f"Feature-generering fejlede: {e}")

    if df_features is None and st.session_state.df_features is not None:
        df_features = st.session_state.df_features

    run_from_main = False
    if df_features is not None:
        df_features = _ensure_ts(df_features)
        df_features = _prepare_indicators(df_features)
        st.session_state.df_features = df_features

        with st.expander("👀 Se data (øverste rækker)", expanded=True):
            st.dataframe(df_features.head(50), width='stretch')

        run_from_main = st.button("Kør backtest (hurtig)", key="run_main")

    if df_features is not None and (run_from_form or run_from_main):
        try:
            with st.spinner("Kører backtest…"):
                if strategy == "RSI":
                    signals = _signals_rsi(
                        df_features,
                        length=rsi_len, low=rsi_low, high=rsi_high,
                        position_mode=position_map[position_mode],
                        regime=regime_map[regime],
                        cooldown=cooldown,
                        min_atr_pct=min_atr_pct,
                    )
                elif strategy == "EMA Cross":
                    signals = _signals_ema(
                        df_features,
                        fast=ema_fast, slow=ema_slow,
                        position_mode=position_map[position_mode],
                        regime=regime_map[regime],
                        cooldown=cooldown,
                        min_atr_pct=min_atr_pct,
                    )
                elif strategy == "MACD":
                    signals = _signals_macd(
                        df_features,
                        fast=macd_fast, slow=macd_slow, signal=macd_signal,
                        position_mode=position_map[position_mode],
                        regime=regime_map[regime],
                        cooldown=cooldown,
                        min_atr_pct=min_atr_pct,
                    )
                else:
                    signals = _signals_ensemble(
                        df_features,
                        rsi_len=rsi_len, rsi_low=rsi_low, rsi_high=rsi_high,
                        ema_fast=ema_fast, ema_slow=ema_slow,
                        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                        position_mode=position_map[position_mode],
                        regime=regime_map[regime],
                        cooldown=cooldown,
                        min_atr_pct=min_atr_pct,
                    )

                trades, balance, metrics = _backtest_and_metrics(df_features, signals)
                tag = f"{symbol}_{timeframe}_{strategy.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                outdir = _save_artifacts(trades, balance, metrics, tag)
                st.session_state.last_results = (trades, balance, metrics, outdir)

            st.toast("✅ Backtest færdig – resultater vist nedenfor", icon="✅")
        except Exception as e:
            st.exception(e)

    if st.session_state.last_results is not None:
        trades, balance, metrics, outdir = st.session_state.last_results

        left, right = st.columns([2, 1])
        with left:
            fig = _plot_equity(balance, title="Equity")
            st.pyplot(fig, clear_figure=True)
        with right:
            st.markdown("#### 📊 Metrics")
            st.json(metrics)
            st.caption(f"Artefakter gemt i: `{outdir}`")

        with st.expander("🔎 Trades (første 200)"):
            st.dataframe(trades.head(200), width='stretch')
        with st.expander("📈 Balance (første 200)"):
            st.dataframe(balance.head(200), width='stretch')

    if df_features is None and st.session_state.last_results is None:
        st.info("Upload/indlæs data ovenfor for at komme i gang.")

def run_streamlit():
    st.set_page_config(page_title="AI Trading – Live & Backtest", page_icon="📈", layout="wide")
    st.title("📈 AI Trading – Live Monitor & Backtest")
    tabs = st.tabs(["Live Monitor", "Backtest"])
    with tabs[0]:
        _render_live_monitor()
    with tabs[1]:
        _render_backtest()

# -------------------------
# CLI fallback
# -------------------------
def run_cli():
    parser = argparse.ArgumentParser(description="GUI/CLI app – backtest simple strategier på features CSV.")
    parser.add_argument("--features", type=str, help="Sti til features-CSV (hvis udeladt forsøges 'latest').", default=None)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--strategy", type=str, choices=["rsi", "ema", "macd", "ensemble"], default="ensemble")
    parser.add_argument("--rsi_len", type=int, default=14)
    parser.add_argument("--rsi_low", type=int, default=45)
    parser.add_argument("--rsi_high", type=int, default=55)
    parser.add_argument("--ema_fast", type=int, default=9)
    parser.add_argument("--ema_slow", type=int, default=21)
    parser.add_argument("--macd_fast", type=int, default=12)
    parser.add_argument("--macd_slow", type=int, default=26)
    parser.add_argument("--macd_signal", type=int, default=9)
    parser.add_argument("--position_mode", type=str, choices=["both", "long_only", "short_only"], default="both")
    parser.add_argument("--regime", type=str, choices=["none", "price_vs_ema200", "ema200_slope"], default="none")
    parser.add_argument("--cooldown", type=int, default=2)
    parser.add_argument("--min_atr_pct", type=float, default=0.0)
    args = parser.parse_args()

    if args.features:
        df = pd.read_csv(args.features, engine="python", on_bad_lines="skip")
    else:
        df = _silent_read(load_features, args.symbol, args.timeframe, version_prefix=None)

    df = _ensure_ts(df)
    df = _prepare_indicators(df)

    if args.strategy == "rsi":
        signals = _signals_rsi(
            df, length=args.rsi_len, low=args.rsi_low, high=args.rsi_high,
            position_mode=args.position_mode, regime=args.regime,
            cooldown=args.cooldown, min_atr_pct=args.min_atr_pct,
        )
    elif args.strategy == "ema":
        signals = _signals_ema(
            df, fast=args.ema_fast, slow=args.ema_slow,
            position_mode=args.position_mode, regime=args.regime,
            cooldown=args.cooldown, min_atr_pct=args.min_atr_pct,
        )
    elif args.strategy == "macd":
        signals = _signals_macd(
            df, fast=args.macd_fast, slow=args.macd_slow, signal=args.macd_signal,
            position_mode=args.position_mode, regime=args.regime,
            cooldown=args.cooldown, min_atr_pct=args.min_atr_pct,
        )
    else:
        signals = _signals_ensemble(
            df,
            rsi_len=args.rsi_len, rsi_low=args.rsi_low, rsi_high=args.rsi_high,
            ema_fast=args.ema_fast, ema_slow=args.ema_slow,
            macd_fast=args.macd_fast, macd_slow=args.macd_slow, macd_signal=args.macd_signal,
            position_mode=args.position_mode, regime=args.regime,
            cooldown=args.cooldown, min_atr_pct=args.min_atr_pct,
        )

    trades, balance, metrics = _backtest_and_metrics(df, signals)

    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    tag = f"{args.symbol}_{args.timeframe}_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = _save_artifacts(trades, balance, metrics, tag)
    print(f"\nArtefakter gemt i: {outdir}")

    fig = _plot_equity(balance, title=f"Equity – {args.strategy.upper()}")
    png_path = Path(outdir) / "equity.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=144)
    print(f"Figur gemt: {png_path}")

if __name__ == "__main__":
    if _HAS_ST and os.environ.get("FORCE_CLI", "").lower() not in ("1", "true", "yes"):
        run_streamlit()
    else:
        run_cli()
