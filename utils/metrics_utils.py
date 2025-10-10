# utils/metrics_utils.py
# -*- coding: utf-8 -*-
"""
Robuste performance-metrikker til analyze-/paper-flow (Fase 4).

Konventioner:
- trades_df['profit'] er afkast pr. afsluttet trade i decimaltal (fx +0.012 = +1.2%).
- 'best_trade'/'worst_trade' rapporteres i procent.
- balance_df['balance'] eller ['equity'] er kontosaldo/equity (float).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------
# Hjælpere (robuste serier)
# ---------------------------
def _to_float_series(x) -> pd.Series:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def _has_cols(df: pd.DataFrame, cols) -> bool:
    return isinstance(df, pd.DataFrame) and all(c in df.columns for c in cols)


def _equity_series(balance_df: pd.DataFrame) -> pd.Series:
    """
    Returnér en robust equity/balance serie:
    - Foretrækker 'balance'
    - Ellers 'equity'
    - Renser NaN/Inf og ffill/bfill
    """
    if isinstance(balance_df, pd.DataFrame):
        for col in ("balance", "equity", "Balance", "Equity"):
            if col in balance_df:
                s = _to_float_series(balance_df[col])
                if s.size >= 1:
                    return s
    return pd.Series(dtype=float)


def _returns_from_equity(eq: pd.Series) -> pd.Series:
    if eq.size <= 1:
        return pd.Series(dtype=float)
    r = eq.pct_change()
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _max_drawdown_pct(eq: pd.Series) -> float:
    """Max drawdown i procent (negativt tal)."""
    if eq.size < 2:
        return 0.0
    roll_max = eq.cummax().replace(0, np.nan)
    dd = (eq - roll_max) / roll_max
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min() * 100.0)  # negativt tal, fx -12.3


def _exit_mask(trades_df: pd.DataFrame) -> pd.Series:
    """
    Mask for afsluttede handler.
    - Hvis 'type' findes: brug kun TP/SL/CLOSE
    - Ellers: brug rækker med gyldig 'profit'
    """
    if "type" in trades_df.columns:
        types = trades_df["type"].astype(str).str.upper()
        return types.isin(["TP", "SL", "CLOSE"])
    return pd.to_numeric(
        trades_df.get("profit", pd.Series(dtype=float)), errors="coerce"
    ).notna()


def _exit_profits(trades_df: pd.DataFrame) -> pd.Series:
    if (
        not isinstance(trades_df, pd.DataFrame)
        or trades_df.empty
        or "profit" not in trades_df.columns
    ):
        return pd.Series(dtype=float)
    mask = _exit_mask(trades_df)
    return (
        pd.to_numeric(trades_df.loc[mask, "profit"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


# ---------------------------
# Dine eksisterende helpers (forbedret robusthed)
# ---------------------------
def max_consecutive_losses(trades_df: pd.DataFrame) -> int:
    """Returnerer længste streak af tab i afsluttede trades ('profit'<0)."""
    pr = _exit_profits(trades_df)
    if pr.empty:
        return 0
    losses = (pr < 0).astype(int)
    if losses.empty:
        return 0
    streak = losses.groupby((losses != losses.shift()).cumsum()).cumsum() * losses
    try:
        val = int(streak.max())
    except Exception:
        val = 0
    return val or 0


def recovery_bars(balance_df: pd.DataFrame) -> int:
    """Hvor mange bars tager det at genvinde tidligere peak efter max drawdown?"""
    eq = _equity_series(balance_df)
    if eq.empty:
        return -1
    peak = eq.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = (eq - peak) / peak.replace(0, np.nan)
    if drawdown.empty:
        return -1
    dd_end = drawdown.idxmin()
    if dd_end is None or (isinstance(dd_end, float) and np.isnan(dd_end)):
        return -1
    prev_peak = float(peak.loc[dd_end])
    after = eq.loc[dd_end:]
    rec_idx = after[after >= prev_peak].first_valid_index()
    if rec_idx is None:
        return -1
    try:
        # positionsbaseret afstand
        return int(eq.index.get_loc(rec_idx) - eq.index.get_loc(dd_end))
    except Exception:
        # fallback hvis index ikke er RangeIndex
        try:
            return int(max(0, after.index.tolist().index(rec_idx)))
        except Exception:
            return -1


def profit_factor(trades_df: pd.DataFrame) -> float:
    """Ratio af samlet gevinst til samlet tab for afsluttede handler. NaN hvis ikke defineret."""
    pr = _exit_profits(trades_df)
    if pr.empty:
        return float("nan")
    gross_profit = float(pr[pr > 0].sum())
    gross_loss = float(-pr[pr < 0].sum())
    if gross_loss == 0.0:
        return float("nan")
    return round(gross_profit / gross_loss, 4)


def sharpe_ratio(
    balance_df: pd.DataFrame, annualization_factor: float = 252.0
) -> float:
    """Sharpe-ratio baseret på pct_change i equity (annualiseret)."""
    eq = _equity_series(balance_df)
    rets = _returns_from_equity(eq)
    if rets.empty:
        return 0.0
    std = float(rets.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return round((float(rets.mean()) / std) * np.sqrt(annualization_factor), 4)


def sortino_ratio(
    balance_df: pd.DataFrame, annualization_factor: float = 252.0
) -> float:
    """Sortino-ratio baseret på pct_change i equity (annualiseret)."""
    eq = _equity_series(balance_df)
    rets = _returns_from_equity(eq)
    if rets.empty:
        return 0.0
    downside = rets[rets < 0]
    denom = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
    if denom == 0.0 or np.isnan(denom):
        return 0.0
    return round((float(rets.mean()) / denom) * np.sqrt(annualization_factor), 4)


def win_rate(trades_df: pd.DataFrame) -> float:
    """Andel af vindende afsluttede handler (profit > 0) i %."""
    pr = _exit_profits(trades_df)
    if pr.empty:
        return 0.0
    total = int((~pr.isna()).sum())
    if total == 0:
        return 0.0
    wins = int((pr > 0).sum())
    return round(100.0 * wins / total, 2)


def best_trade(trades_df: pd.DataFrame) -> float:
    """Største profit (procent) blandt afsluttede handler (profit i decimaltal)."""
    pr = _exit_profits(trades_df)
    return round(float(pr.max()) * 100.0, 2) if not pr.empty else 0.0


def worst_trade(trades_df: pd.DataFrame) -> float:
    """Største tab (procent) blandt afsluttede handler (profit i decimaltal)."""
    pr = _exit_profits(trades_df)
    return round(float(pr.min()) * 100.0, 2) if not pr.empty else 0.0


# ---------------------------
# Hoved-API kaldt af engine/backtest
# ---------------------------
def advanced_performance_metrics(
    trades_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    initial_balance: float = 1000.0,
) -> Dict[str, float]:
    """
    Returnerer dictionary med vigtige performance-metrics – klar til monitoring/CI.
    Robust overfor manglende/konstante data.
    """

    # --- Equity & afkast ---
    eq = _equity_series(balance_df)
    if not eq.empty and eq.iloc[0] > 0:
        init_eq = float(eq.iloc[0])
    else:
        init_eq = float(initial_balance)

    if not eq.empty and len(eq) >= 2:
        pnl_abs = float(eq.iloc[-1] - eq.iloc[0])
        ret_total = pnl_abs / max(eq.iloc[0], 1.0)
        dd_pct = _max_drawdown_pct(eq)  # negativ
    else:
        ret_total = 0.0
        dd_pct = 0.0

    # --- Trades (kun afslutninger) ---
    pr = _exit_profits(trades_df)
    num_trades = int(len(pr)) if not pr.empty else 0
    wr = win_rate(trades_df) if num_trades > 0 else 0.0
    bt = best_trade(trades_df) if num_trades > 0 else 0.0
    wt = worst_trade(trades_df) if num_trades > 0 else 0.0
    pf = profit_factor(trades_df) if num_trades > 0 else float("nan")

    # --- Sharpe / Sortino ---
    sharpe = sharpe_ratio(balance_df) if not eq.empty else 0.0
    sortino = sortino_ratio(balance_df) if not eq.empty else 0.0

    out = {
        "profit_pct": round(ret_total * 100.0, 4),
        "max_drawdown": round(dd_pct, 4),  # negativ i %
        "win_rate": wr,  # i %
        "best_trade": bt,  # i %
        "worst_trade": wt,  # i %
        "max_consec_losses": int(
            max_consecutive_losses(trades_df) if not pr.empty else 0
        ),
        "recovery_bars": int(recovery_bars(balance_df)),
        "profit_factor": float(pf) if pf == pf else float("nan"),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "num_trades": int(num_trades),
    }
    return out
