# utils/metrics_utils.py

import numpy as np
import pandas as pd

def max_consecutive_losses(trades_df):
    """Returnerer længste streak af tab i trades_df['profit'].""" 
    if "profit" not in trades_df or trades_df.empty:
        return 0
    losses = (trades_df['profit'] < 0).astype(int)
    return (losses.groupby((losses != losses.shift()).cumsum()).cumsum() * losses).max() or 0

def recovery_bars(balance_df):
    """Hvor mange bars tager det at genvinde tidligere peak efter max drawdown?"""
    if balance_df.empty or "balance" not in balance_df:
        return -1
    peak = balance_df["balance"].cummax()
    drawdown = (balance_df["balance"] - peak) / peak
    dd_end = drawdown.idxmin() if not drawdown.empty else None
    if dd_end is not None and not np.isnan(dd_end):
        balance_after_dd = balance_df.loc[dd_end:]['balance']
        prev_peak = peak.loc[dd_end]
        recovery_idx = balance_after_dd[balance_after_dd >= prev_peak].first_valid_index()
        if recovery_idx is not None:
            return balance_df.index.get_loc(recovery_idx) - balance_df.index.get_loc(dd_end)
    return -1

def profit_factor(trades_df):
    """Ratio af samlet gevinst til samlet tab."""
    if "profit" not in trades_df or trades_df.empty:
        return "N/A"
    gross_profit = trades_df.loc[trades_df["profit"] > 0, "profit"].sum()
    gross_loss = -trades_df.loc[trades_df["profit"] < 0, "profit"].sum()
    if gross_loss == 0:
        return np.nan
    return round(gross_profit / gross_loss, 2)

def sharpe_ratio(balance_df, annualization_factor=252):
    """Sharpe-ratio baseret på pct_change i balance."""
    if balance_df.empty or "balance" not in balance_df:
        return 0
    returns = balance_df["balance"].pct_change().dropna()
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0
    return round((returns.mean() / std) * np.sqrt(annualization_factor), 2)

def sortino_ratio(balance_df, annualization_factor=252):
    """Sortino-ratio baseret på pct_change i balance."""
    if balance_df.empty or "balance" not in balance_df:
        return 0
    returns = balance_df["balance"].pct_change().dropna()
    downside = returns[returns < 0]
    std = downside.std()
    if std == 0 or np.isnan(std):
        return 0
    return round((returns.mean() / std) * np.sqrt(annualization_factor), 2)

def win_rate(trades_df):
    """Andel af vindende handler (TP eller profit > 0)."""
    if trades_df.empty or "profit" not in trades_df:
        return 0.0
    wins = trades_df.loc[trades_df["profit"] > 0]
    return round(100 * len(wins) / len(trades_df), 2) if len(trades_df) > 0 else 0.0

def best_trade(trades_df):
    """Største profit (procent) i trades_df."""
    if trades_df.empty or "profit" not in trades_df:
        return 0.0
    return round(trades_df["profit"].max() * 100, 2)

def worst_trade(trades_df):
    """Største tab (procent) i trades_df."""
    if trades_df.empty or "profit" not in trades_df:
        return 0.0
    return round(trades_df["profit"].min() * 100, 2)

def advanced_performance_metrics(trades_df, balance_df, initial_balance=1000):
    """
    Returnerer dictionary med alle vigtige performance-metrics – klar til monitoring, Telegram, CI m.m.
    """
    # Default hvis ingen handler
    if trades_df.empty or "profit" not in trades_df or len(trades_df["profit"]) == 0:
        return {
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "max_consec_losses": 0,
            "recovery_bars": -1,
            "profit_factor": "N/A",
            "sharpe": 0,
            "sortino": 0,
        }
    profit_pct = (balance_df['balance'].iloc[-1] - initial_balance) / initial_balance * 100
    peak = balance_df['balance'].cummax()
    drawdown = (balance_df['balance'] - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    return {
        "profit_pct": round(profit_pct, 2),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": win_rate(trades_df),
        "best_trade": best_trade(trades_df),
        "worst_trade": worst_trade(trades_df),
        "max_consec_losses": int(max_consecutive_losses(trades_df)),
        "recovery_bars": int(recovery_bars(balance_df)),
        "profit_factor": profit_factor(trades_df),
        "sharpe": sharpe_ratio(balance_df),
        "sortino": sortino_ratio(balance_df),
    }
