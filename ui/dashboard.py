# ui/dashboard.py
"""
Streamlit GUI v1 for paper/live overvågning
- KPI-kort: Total PnL %, Win-rate, Max Drawdown, Antal handler
- Equity-graf fra logs/equity.csv (tolerant ift. kolonnenavne)
- Seneste handler (fills.csv) og signaler (signals.csv)
- Auto-refresh uden ekstra afhængigheder (JS-reload)
- Parametre i sidebar: LOG_DIR, Auto-refresh (sek), Max rows

NYT i denne version:
- Automatisk detektion af PnL-kolonne til Win-rate (fx 'pnl', 'pnl_realized', 'pnl_realised', 'profit', ...)

Kør:  streamlit run ui/dashboard.py
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Sidedesign
# -------------------------------------------------
st.set_page_config(page_title="TradeOps Dashboard", layout="wide")
st.title("📊 TradeOps Dashboard")

# -------------------------------------------------
# Sidebar-parametre
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Indstillinger")
    default_log_dir = os.getenv("LOG_DIR", "logs")
    log_dir = Path(st.text_input("LOG_DIR", value=default_log_dir))
    auto_refresh_sec = st.number_input("Auto-refresh (sek)", min_value=0, max_value=600, value=15, step=5)
    max_rows = st.number_input("Max rows (tabeller)", min_value=100, max_value=100_000, value=2_000, step=100)
    st.caption("Tip: Peg på alternativ log-mappe uden kodeændringer.")

def _inject_auto_refresh(seconds: int) -> None:
    """Reload hele siden i browseren efter N sekunder (stabilt på Windows)."""
    if seconds and seconds > 0:
        # Simpelt JS-reload; undgår afhængighed af st_autorefresh-versioner
        st.markdown(
            f"<script>setTimeout(function() {{ window.location.reload(); }}, {int(seconds)*1000});</script>",
            unsafe_allow_html=True,
        )

_inject_auto_refresh(int(auto_refresh_sec))

# -------------------------------------------------
# Sikker CSV-læsning (med cache + retries)
# -------------------------------------------------
@st.cache_data(ttl=5)
def read_csv_safe(path: Path, parse_dates: Optional[list] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    last_err: Exception | None = None
    for _ in range(3):
        try:
            return pd.read_csv(
                path,
                on_bad_lines="skip",
                low_memory=False,
                parse_dates=parse_dates,
                nrows=nrows,
            )
        except Exception as e:  # midlertidig fil-lock/skrivning → prøv igen
            last_err = e
            time.sleep(0.1)
    st.warning(f"Kunne ikke læse {path.name}: {last_err}")
    return pd.DataFrame()

# -------------------------------------------------
# Indlæs data
# -------------------------------------------------
EQUITY_CSV = log_dir / "equity.csv"
FILLS_CSV = log_dir / "fills.csv"
SIGNALS_CSV = log_dir / "signals.csv"
ALERTS_LOG = log_dir / "telegram_log.txt"

# equity.csv: forventer første kolonne ~ timestamp, anden ~ equity (heuristik)
equity_df = read_csv_safe(EQUITY_CSV, parse_dates=[0])
if not equity_df.empty:
    equity_df.columns = [str(c).strip().lower() for c in equity_df.columns]
    if "timestamp" not in equity_df.columns:
        equity_df.rename(columns={equity_df.columns[0]: "timestamp"}, inplace=True)
    if "equity" not in equity_df.columns and len(equity_df.columns) >= 2:
        equity_df.rename(columns={equity_df.columns[1]: "equity"}, inplace=True)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], errors="coerce")
    # Fjern rækker uden timestamp/equity
    equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")
    equity_df = equity_df.dropna(subset=["timestamp", "equity"])

fills_df = read_csv_safe(FILLS_CSV)
if not fills_df.empty:
    fills_df.columns = [str(c).strip().lower() for c in fills_df.columns]

signals_df = read_csv_safe(SIGNALS_CSV)
if not signals_df.empty:
    signals_df.columns = [str(c).strip().lower() for c in signals_df.columns]

# Klip tabeller for snappy UI
if not fills_df.empty and len(fills_df) > max_rows:
    fills_df = fills_df.tail(max_rows)
if not signals_df.empty and len(signals_df) > max_rows:
    signals_df = signals_df.tail(max_rows)

# -------------------------------------------------
# KPI’er
# -------------------------------------------------
def _detect_pnl_col(df: pd.DataFrame) -> Optional[str]:
    """Find bedste match til PnL-kolonne (win-rate)."""
    if df is None or df.empty:
        return None
    candidates = [
        "pnl", "pnl_realized", "pnl_realised", "pnl_total", "pnl_net",
        "profit", "profit_net", "pl", "realized_pnl", "realised_pnl"
    ]
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # grov fallback: find første kolonne der ligner positiv/negativ PnL
    for c in df.columns:
        if any(k in c for k in ["pnl", "profit", "pl"]):
            return c
    return None


def compute_kpis(equity: pd.DataFrame, fills: pd.DataFrame) -> Tuple[float, float, float, int]:
    pnl_pct = 0.0
    win_rate = 0.0
    dd_pct = 0.0
    n_trades = len(fills) if not fills.empty else 0

    # Total PnL% + Max Drawdown
    if not equity.empty and "equity" in equity.columns:
        eq = pd.to_numeric(equity["equity"], errors="coerce").dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0.0:
            pnl_pct = (float(eq.iloc[-1]) / float(eq.iloc[0]) - 1.0) * 100.0
        if not eq.empty:
            roll_max = eq.cummax()
            drawdown = (eq / roll_max - 1.0) * 100.0
            dd_pct = float(drawdown.min()) if not drawdown.empty else 0.0

    # Win-rate via auto-detekteret PnL-kolonne
    if not fills.empty:
        pnl_col = _detect_pnl_col(fills)
        if pnl_col:
            pnl_series = pd.to_numeric(fills[pnl_col], errors="coerce")
            wins = (pnl_series > 0).sum()
            total = pnl_series.notna().sum()
            if total > 0:
                win_rate = wins / total * 100.0

    return pnl_pct, win_rate, dd_pct, n_trades


pnl_pct, win_rate, dd_pct, n_trades = compute_kpis(equity_df, fills_df)

# KPI-kort
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total PnL %", f"{pnl_pct:,.2f}%")
c2.metric("Win-rate", f"{win_rate:,.1f}%")
c3.metric("Max DD", f"{dd_pct:,.2f}%")
c4.metric("Antal handler", f"{n_trades}")

# -------------------------------------------------
# Equity-graf
# -------------------------------------------------
st.subheader("Equity-kurve")
if equity_df.empty:
    st.info(f"Ingen data endnu i {EQUITY_CSV}")
else:
    plot_df = equity_df[["timestamp", "equity"]].copy().sort_values("timestamp").set_index("timestamp")
    st.line_chart(plot_df)

# -------------------------------------------------
# Seneste handler & signaler
# -------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Seneste handler (fills.csv)")
    if fills_df.empty:
        st.info(f"Ingen data endnu i {FILLS_CSV}")
    else:
        st.dataframe(fills_df, use_container_width=True)
        # Eksport af nuværende visning
        buf = io.StringIO()
        fills_df.to_csv(buf, index=False)
        st.download_button(
            label="Download handler (CSV)",
            data=buf.getvalue(),
            file_name="fills_filtered.csv",
            mime="text/csv",
        )

with right:
    st.subheader("Seneste signaler (signals.csv)")
    if signals_df.empty:
        st.info(f"Ingen data endnu i {SIGNALS_CSV}")
    else:
        st.dataframe(signals_df, use_container_width=True)

# -------------------------------------------------
# Alerts / fejl-log (valgfrit)
# -------------------------------------------------
if ALERTS_LOG.exists():
    st.subheader("Alerts / Fejl (seneste 200 linjer)")
    try:
        text = ALERTS_LOG.read_text(encoding="utf-8", errors="ignore")
        tail = "\n".join(text.splitlines()[-200:])
        st.code(tail, language="text")
    except Exception as e:
        st.warning(f"Kan ikke læse {ALERTS_LOG.name}: {e}")

st.caption("© Lyra-TradeOps — GUI opdateres automatisk uden at blokere engine.")
