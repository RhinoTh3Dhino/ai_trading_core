# gui/app.py
"""
AI Trading GUI – Live Monitor + Backtest (Dag 6–8)

NYT (Dag 6–8):
- Fanen “Live Monitor”: henter status/metrics/equity fra API (fallback: CSV/JSON),
  viser statuskort (seneste kørsel, win-rate 7d, drawdown), equity-kurve og live-signaler.
- API-base kan sættes via ENV: API_BASE_URL eller i UI.
- Auto-refresh via st_autorefresh (valgfri pakke) – fallback hvis ikke tilgængelig.

BEVARER (fra tidligere version):
- Backtest-fanen: indlæs/generér features, kør RSI/EMA/MACD/Ensemble, globale filtre,
  metrics, equity-plot og artefakt-eksport til outputs/gui/<timestamp>/.
- Session State, caching, CLI-fallback.

WHY (teknisk/strategisk):
- Tabs adskiller tydeligt driftsovervågning (paper/live) fra forsknings-backtests.
- API→GUI med CSV-fallback giver robusthed og hurtig SaaS-parathed.
- Ingen breaking changes i core-funktioner; kun UI-omlægning for klarere brugerrejse.
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

# Robust backend for plots når vi ikke har display
os.environ.setdefault("MPLBACKEND", "Agg")

# --- bootstrap PYTHONPATH så imports virker fra 'gui/' ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]  # repo-roden
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ekstra: requests er valgfrit (GUI falder tilbage til lokale filer hvis ikke installeret)
try:  # pragma: no cover
    import requests
    _HAS_REQ = True
except Exception:  # pragma: no cover
    requests = None
    _HAS_REQ = False

# Auto-refresh: brug ekstern helper hvis den findes; ellers no-op fallback
try:  # pragma: no cover
    # pip install streamlit-autorefresh
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    def st_autorefresh(*_, **__):
        # Fallback: gør ingenting; brugeren kan opdatere manuelt
        return 0

# Projekt-imports
from utils.project_path import PROJECT_ROOT
from backtest.backtest import run_backtest
from utils.metrics_utils import advanced_performance_metrics
from features.features_pipeline import generate_features, load_features

# Prøv at importere de eksisterende strategier – ellers falder vi tilbage
try:
    from strategies.rsi_strategy import rsi_rule_based_signals as _rsi_signals_lib  # type: ignore
except Exception:  # pragma: no cover
    _rsi_signals_lib = None

try:
    from strategies.ema_cross_strategy import ema_cross_signals as _ema_signals_lib  # type: ignore
except Exception:  # pragma: no cover
    _ema_signals_lib = None

try:
    from strategies.macd_strategy import macd_cross_signals as _macd_signals_lib  # type: ignore
except Exception:  # pragma: no cover
    _macd_signals_lib = None

# Streamlit er valgfri – vi kører kun GUI-delen hvis det er tilgængeligt
try:  # pragma: no cover
    import streamlit as st
    _HAS_ST = True
except Exception:  # pragma: no cover
    st = None
    _HAS_ST = False


# -------------------------
# Små hjælpefunktioner
# -------------------------
ROOT = Path(PROJECT_ROOT)
LOGS = ROOT / "logs"
API_DIR = ROOT / "api"

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Sørg for en 'timestamp' kolonne og sorter."""
    out = df.copy()
    if "timestamp" not in out.columns:
        for cand in ("datetime", "Timestamp"):
            if cand in out.columns:
                out = out.rename(columns={cand: "timestamp"})
                break
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
    position_mode: str = "both",            # "both" | "long_only" | "short_only"
    regime: str = "none",                   # "none" | "price_vs_ema200" | "ema200_slope"
    cooldown: int = 0,                      # bars to wait after flip
    min_atr_pct: float = 0.0,               # require ATR% >= this for entries
) -> np.ndarray:
    sig = raw_sig.astype(int).copy()
    n = len(sig)
    if n == 0:
        return sig

    # Positionstilstand
    if position_mode == "long_only":
        sig = (sig > 0).astype(int)
    elif position_mode == "short_only":
        # I denne simple 0/1-model er "short" et flip til 0; vi kan derfor bare flade longs
        sig = np.zeros_like(sig, dtype=int)

    # Regime-filter
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
            # Undertryk short-flip hvis short-regime ikke er ok
            if i > 0 and raw_sig[i] == 0 and raw_sig[i - 1] == 1 and not bool(short_ok.iloc[i]):
                sig[i] = 1

    # ATR%-filter (kun tillad flip når volatiliteten er "tilstrækkelig")
    if min_atr_pct and "atr_14" in df.columns:
        atr_pct = (df["atr_14"].astype(float) / df["close"].astype(float)).fillna(0.0) * 100.0
        for i in range(1, n):
            flipped = (sig[i] != sig[i - 1])
            if flipped and atr_pct.iloc[i] < float(min_atr_pct):
                sig[i] = sig[i - 1]

    # Cooldown (debounce) – kræv N bar mellem flips
    if cooldown and cooldown > 0:
        last_flip = 0
        for i in range(1, n):
            if sig[i] != sig[i - 1]:
                if (i - last_flip) <= cooldown:
                    sig[i] = sig[i - 1]
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
    raw = np.where(rs > high, 0, np.where(rs < low, 1, 0)).astype(int)  # contrarian
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
    """Kald en funktion men undertryk stdout/prints (fjerner støj i terminalen)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def _silent_run_backtest(df: pd.DataFrame, signals: np.ndarray):
    """Backtest uden prints til stdout."""
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
if _HAS_ST:  # pragma: no cover
    @st.cache_data(ttl=2.0, show_spinner=False)
    def _http_get_json(url: str):
        if not _HAS_REQ:
            return None, "requests ikke installeret"
        try:
            r = requests.get(url, timeout=2.0)
            if r.ok:
                return r.json(), None
            return None, f"HTTP {r.status_code}"
        except Exception as e:
            return None, str(e)

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_status(api_base: str) -> Tuple[Dict, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/status")
        if data is not None:
            return data, None
        # fallback – minimal status
        return {"last_run_ts": datetime.utcnow().isoformat(), "mode": "paper", "win_rate_7d": 0.0, "drawdown_pct": 0.0}, err

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_equity(api_base: str) -> Tuple[pd.DataFrame, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/equity")
        if data is not None:
            return pd.DataFrame(data), None
        p = LOGS / "equity.csv"
        if p.exists():
            return pd.read_csv(p)[["date", "equity"]], f"API fejl: {err}. Viser CSV."
        return pd.DataFrame(columns=["date", "equity"]), "Ingen equity-data."

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_metrics(api_base: str, limit: int = 30) -> Tuple[pd.DataFrame, Optional[str]]:
        data, err = _http_get_json(f"{api_base}/metrics/daily?limit={limit}")
        if data is not None:
            return pd.DataFrame(data), None
        p = LOGS / "daily_metrics.csv"
        if p.exists():
            return pd.read_csv(p).tail(limit), f"API fejl: {err}. Viser CSV."
        return pd.DataFrame(), "Ingen metrikker."

    @st.cache_data(ttl=2.0, show_spinner=False)
    def lm_load_signals(api_base: str) -> Tuple[List[Dict], Optional[str]]:
        data, err = _http_get_json(f"{api_base}/signals/latest")
        if data:
            # API kan returnere et objekt (seneste) eller en liste; normalisér til liste
            if isinstance(data, dict):
                return [data], None
            if isinstance(data, list):
                return data[-20:], None
        # fallback: lokal mock
        p = API_DIR / "sim_signals.json"
        if p.exists():
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
                return arr[-20:], "API utilgængelig — viser lokale mock-signaler."
            except Exception as e:
                return [], f"Kan ikke læse mock-signaler: {e}"
        return [], "Ingen signaler."


# -------------------------
# Streamlit helpers (cache) – backtest
# -------------------------
if _HAS_ST:  # pragma: no cover
    @st.cache_data(show_spinner=False)
    def _cached_latest(symbol: str, timeframe: str) -> pd.DataFrame:
        return _silent_read(load_features, symbol, timeframe, version_prefix=None)


# -------------------------
# Streamlit GUI
# -------------------------
def _render_live_monitor():  # pragma: no cover
    st.subheader("📡 Live Monitor (Paper Trading)")
    api_default = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    api_base = st.text_input("API base URL", value=st.session_state.get("api_base", api_default))
    st.session_state["api_base"] = api_base
    interval = st.slider("Auto-refresh (sek.)", 1, 10, 3, 1)
    st_autorefresh(interval=interval * 1000, key="live_refresh")

    # Statuskort
    status, status_err = lm_load_status(api_base)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Seneste kørsel (UTC)", value=str(status.get("last_run_ts", "")).replace("T", " ").split(".")[0])
    with c2:
        st.metric("Win-rate (7d)", f'{float(status.get("win_rate_7d", 0.0)):.2f}%')
    with c3:
        st.metric("Drawdown", f'{float(status.get("drawdown_pct", 0.0)):.2f}%')
    if status_err:
        st.info(f"Status fra fallback: {status_err}")

    # Live signaler
    st.markdown("### Live signaler")
    sigs, sig_err = lm_load_signals(api_base)
    if sig_err:
        st.info(sig_err)
    if sigs:
        df_sig = pd.DataFrame(sigs).copy()
        if "timestamp" in df_sig.columns:
            df_sig["ts"] = pd.to_datetime(df_sig["timestamp"], unit="s", errors="coerce")
            order_cols = [c for c in ["ts", "symbol", "side", "confidence", "price", "regime"] if c in df_sig.columns]
        else:
            order_cols = [c for c in ["symbol", "side", "confidence", "price", "regime"] if c in df_sig.columns]
        st.dataframe(df_sig[order_cols].sort_values(order_cols[0], ascending=False) if order_cols else df_sig, use_container_width=True)
    else:
        st.warning("Ingen signaler.")

    # Equity-kurve
    st.markdown("### Equity-kurve")
    df_eq, eq_err = lm_load_equity(api_base)
    if eq_err:
        st.info(eq_err)
    if not df_eq.empty:
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(df_eq["date"]), df_eq["equity"])
        ax.set_xlabel("Dato"); ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("Ingen equity-data.")

    # Daglige metrikker
    st.markdown("### Daglige metrikker (seneste 30 dage)")
    df_m, m_err = lm_load_metrics(api_base, 30)
    if m_err:
        st.info(m_err)
    if not df_m.empty:
        st.dataframe(df_m, use_container_width=True)
    else:
        st.warning("Ingen metrikker.")


def _render_backtest():  # pragma: no cover
    st.subheader("🧪 Backtest")
    # Init session_state (persist over reruns)
    st.session_state.setdefault("df_features", None)      # seneste feature-DF
    st.session_state.setdefault("last_results", None)     # (trades, balance, metrics, outdir)

    # --- Kontrolpanel i en FORM i hovedområdet (ikke sidebar) ---
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
                                    help="ATR14 / Close * 100. Fx 0.5% på 1h kan skære meget støj fra.")

        run_from_form = st.form_submit_button("Kør backtest", use_container_width=True)

    # map UI strings til interne værdier
    position_map = {"Long & Short": "both", "Kun Long": "long_only", "Kun Short": "short_only"}
    regime_map = {"Ingen": "none", "Pris vs EMA200": "price_vs_ema200", "EMA200-slope": "ema200_slope"}

    df_features: Optional[pd.DataFrame] = None

    # --- Data input ---
    if mode == "Indlæs features (anbefalet)":
        uploaded = st.file_uploader(
            "Upload en features-CSV (eller lad være og indlæs 'latest' fra outputs/feature_data)",
            type=["csv"],
        )
        if uploaded is not None:
            df_features = pd.read_csv(uploaded)
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
            "Upload rå OHLCV CSV (kræver mindst: timestamp, open, high, low, close, volume)",
            type=["csv"],
        )
        if raw is not None:
            raw_df = pd.read_csv(raw)
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

    # Brug gemt DF hvis ingen ny i dette rerun
    if df_features is None and st.session_state.df_features is not None:
        df_features = st.session_state.df_features

    # Vis data + ekstra "Kør backtest"-knap
    run_from_main = False
    if df_features is not None:
        df_features = _ensure_ts(df_features)
        df_features = _prepare_indicators(df_features)
        st.session_state.df_features = df_features  # hold renset + forberedt version

        with st.expander("👀 Se data (øverste rækker)", expanded=True):
            st.dataframe(df_features.head(50), use_container_width=True)

        run_from_main = st.button("Kør backtest (hurtig)", key="run_main", use_container_width=True)

    # Kør backtest hvis nogen af knapperne blev trykket
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

    # Vis seneste resultater (persisterer over reruns)
    if st.session_state.last_results is not None:
        trades, balance, metrics, outdir = st.session_state.last_results

        left, right = st.columns([2, 1])
        with left:
            fig = _plot_equity(balance, title="Equity")
            st.pyplot(fig, use_container_width=True)
        with right:
            st.markdown("#### 📊 Metrics")
            st.json(metrics)
            st.caption(f"Artefakter gemt i: `{outdir}`")

        with st.expander("🔎 Trades (første 200)"):
            st.dataframe(trades.head(200), use_container_width=True)
        with st.expander("📈 Balance (første 200)"):
            st.dataframe(balance.head(200), use_container_width=True)

    if df_features is None and st.session_state.last_results is None:
        st.info("Upload/indlæs data ovenfor for at komme i gang.")


def run_streamlit():  # pragma: no cover
    st.set_page_config(page_title="AI Trading – Live & Backtest", page_icon="📈", layout="wide")
    st.title("📈 AI Trading – Live Monitor & Backtest")

    tabs = st.tabs(["Live Monitor", "Backtest"])
    with tabs[0]:
        _render_live_monitor()
    with tabs[1]:
        _render_backtest()


# -------------------------
# CLI fallback (uden Streamlit)
# -------------------------
def run_cli():
    parser = argparse.ArgumentParser(description="GUI/CLI app – backtest simple strategier på features CSV.")
    parser.add_argument("--features", type=str, help="Sti til features-CSV (hvis udeladt forsøges 'latest').", default=None)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--strategy", type=str, choices=["rsi", "ema", "macd", "ensemble"], default="ensemble")
    # Tunings / filtre (samme defaults som GUI)
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

    # Indlæs features (stille)
    if args.features:
        df = pd.read_csv(args.features)
    else:
        df = _silent_read(load_features, args.symbol, args.timeframe, version_prefix=None)

    df = _ensure_ts(df)
    df = _prepare_indicators(df)

    # Strategi
    if args.strategy == "rsi":
        signals = _signals_rsi(
            df,
            length=args.rsi_len, low=args.rsi_low, high=args.rsi_high,
            position_mode=args.position_mode, regime=args.regime,
            cooldown=args.cooldown, min_atr_pct=args.min_atr_pct,
        )
    elif args.strategy == "ema":
        signals = _signals_ema(
            df,
            fast=args.ema_fast, slow=args.ema_slow,
            position_mode=args.position_mode, regime=args.regime,
            cooldown=args.cooldown, min_atr_pct=args.min_atr_pct,
        )
    elif args.strategy == "macd":
        signals = _signals_macd(
            df,
            fast=args.macd_fast, slow=args.macd_slow, signal=args.macd_signal,
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

    # Print kort resume
    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Gem artefakter
    tag = f"{args.symbol}_{args.timeframe}_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = _save_artifacts(trades, balance, metrics, tag)
    print(f"\nArtefakter gemt i: {outdir}")

    # Gem en equity-PNG for nem deling
    fig = _plot_equity(balance, title=f"Equity – {args.strategy.upper()}")
    png_path = Path(outdir) / "equity.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=144)
    print(f"Figur gemt: {png_path}")


if __name__ == "__main__":  # pragma: no cover
    if _HAS_ST and os.environ.get("FORCE_CLI", "").lower() not in ("1", "true", "yes"):
        run_streamlit()
    else:
        run_cli()
