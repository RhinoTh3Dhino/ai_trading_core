# utils/performance.py
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    sharpe = excess.mean() / (returns.std() + 1e-9) * np.sqrt(periods_per_year)
    return sharpe

def calculate_sortino_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    returns = pd.Series(equity_curve).pct_change().dropna()
    neg_returns = returns[returns < 0]
    downside = neg_returns.std()
    if downside == 0 or len(returns) < 2:
        return 0.0
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    return excess / (downside + 1e-9) * np.sqrt(periods_per_year)

def calculate_max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min() if len(drawdown) > 0 else 0.0

def calculate_winrate(trades_df):
    if 'pnl_%' not in trades_df or len(trades_df) == 0:
        return np.nan
    wins = trades_df['pnl_%'] > 0
    return wins.sum() / len(trades_df) * 100

def calculate_profit_factor(trades_df):
    if 'pnl_%' not in trades_df or len(trades_df) == 0:
        return np.nan
    profits = trades_df[trades_df['pnl_%'] > 0]['pnl_%'].sum()
    losses = -trades_df[trades_df['pnl_%'] < 0]['pnl_%'].sum()
    return profits / losses if losses > 0 else np.nan

def calculate_expectancy(trades_df):
    if 'pnl_%' not in trades_df or len(trades_df) == 0:
        return np.nan
    return trades_df['pnl_%'].mean()

def calculate_profit(equity_curve):
    """Robust udgave – altid brug iloc hvis muligt for Pandas Series, ellers fallback til standard index."""
    if hasattr(equity_curve, "iloc"):
        if len(equity_curve) < 2:
            return 0.0, 0.0
        start = equity_curve.iloc[0]
        end = equity_curve.iloc[-1]
    else:
        if len(equity_curve) < 2:
            return 0.0, 0.0
        start = equity_curve[0]
        end = equity_curve[-1]
    abs_profit = end - start
    pct_profit = (end / start - 1) * 100 if start != 0 else np.nan
    return abs_profit, pct_profit

def calculate_trade_stats(trades_df):
    stats = {
        "total_trades": len(trades_df),
        "win_rate": calculate_winrate(trades_df),
        "expectancy": calculate_expectancy(trades_df),
        "profit_factor": calculate_profit_factor(trades_df),
        "avg_pnl": trades_df['pnl_%'].mean() if 'pnl_%' in trades_df and len(trades_df) > 0 else np.nan,
        "best_trade": trades_df['pnl_%'].max() if 'pnl_%' in trades_df and len(trades_df) > 0 else np.nan,
        "worst_trade": trades_df['pnl_%'].min() if 'pnl_%' in trades_df and len(trades_df) > 0 else np.nan,
        "median_pnl": trades_df['pnl_%'].median() if 'pnl_%' in trades_df and len(trades_df) > 0 else np.nan,
        "num_wins": (trades_df['pnl_%'] > 0).sum() if 'pnl_%' in trades_df and len(trades_df) > 0 else np.nan,
        "num_losses": (trades_df['pnl_%'] < 0).sum() if 'pnl_%' in trades_df and len(trades_df) > 0 else np.nan,
    }
    return stats

def print_performance_report(equity_curve, trades_df):
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    abs_profit, pct_profit = calculate_profit(equity_curve)
    stats = calculate_trade_stats(trades_df)

    print("===== PERFORMANCE REPORT =====")
    print(f"Sharpe ratio   : {sharpe:.2f}")
    print(f"Sortino ratio  : {sortino:.2f}")
    print(f"Max Drawdown   : {max_dd:.2%}")
    print(f"Profit         : {abs_profit:.2f} ({pct_profit:.2f}%)")
    print(f"Win-rate       : {stats['win_rate']:.1f}%")
    print(f"Profit factor  : {stats['profit_factor']:.2f}")
    print(f"Expectancy     : {stats['expectancy']:.2f}%")
    print(f"Antal handler  : {stats['total_trades']}")
    print(f"Bedste trade   : {stats['best_trade']:.2f}%")
    print(f"Værste trade   : {stats['worst_trade']:.2f}%")
    print("==============================")

# --------- GRID SEARCH OG ENSEMBLE/ML READY ---------

def calculate_performance_metrics(equity_curve, trades_df):
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    abs_profit, pct_profit = calculate_profit(equity_curve)
    stats = calculate_trade_stats(trades_df)

    # Robust final_balance (virker både for liste, array, Series)
    if hasattr(equity_curve, "iloc"):
        final_balance = equity_curve.iloc[-1] if len(equity_curve) > 0 else np.nan
    elif hasattr(equity_curve, "__getitem__"):
        final_balance = equity_curve[-1] if len(equity_curve) > 0 else np.nan
    else:
        final_balance = np.nan

    out = {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "final_balance": final_balance,
        "abs_profit": abs_profit,
        "pct_profit": pct_profit,
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "expectancy": stats["expectancy"],
        "total_trades": stats["total_trades"],
        "best_trade": stats["best_trade"],
        "worst_trade": stats["worst_trade"]
    }
    return out
