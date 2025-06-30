# utils/performance.py
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    """
    Beregn Sharpe ratio for equity curve (daglig, 1h, etc.).
    """
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    sharpe = excess.mean() / returns.std() * np.sqrt(periods_per_year)
    return sharpe

def calculate_sortino_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    """
    Sortino Ratio: Sharpe kun på negative afvigelser ("downside risk").
    """
    returns = pd.Series(equity_curve).pct_change().dropna()
    neg_returns = returns[returns < 0]
    downside = neg_returns.std()
    if downside == 0 or len(returns) < 2:
        return 0.0
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    return excess / downside * np.sqrt(periods_per_year)

def calculate_max_drawdown(equity_curve):
    """
    Beregn max drawdown (procent, negativ).
    """
    equity = np.array(equity_curve)
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_winrate(trades_df):
    """
    Beregn win-rate ud fra trades_df med kolonnen 'pnl_%'.
    """
    if 'pnl_%' not in trades_df:
        return np.nan
    wins = trades_df['pnl_%'] > 0
    return wins.sum() / len(trades_df) * 100 if len(trades_df) > 0 else np.nan

def calculate_profit_factor(trades_df):
    """
    Profit factor: sum af positive trades / sum af negative trades.
    """
    if 'pnl_%' not in trades_df:
        return np.nan
    profits = trades_df[trades_df['pnl_%'] > 0]['pnl_%'].sum()
    losses = -trades_df[trades_df['pnl_%'] < 0]['pnl_%'].sum()
    return profits / losses if losses > 0 else np.nan

def calculate_expectancy(trades_df):
    """
    Forventet afkast per trade (gennemsnit).
    """
    if 'pnl_%' not in trades_df:
        return np.nan
    return trades_df['pnl_%'].mean()

def calculate_trade_stats(trades_df):
    """
    Returnér dictionary med alle nøgle-metrics for trades_df.
    """
    stats = {
        "total_trades": len(trades_df),
        "win_rate": calculate_winrate(trades_df),
        "expectancy": calculate_expectancy(trades_df),
        "profit_factor": calculate_profit_factor(trades_df),
        "avg_pnl": trades_df['pnl_%'].mean() if 'pnl_%' in trades_df else np.nan,
        "best_trade": trades_df['pnl_%'].max() if 'pnl_%' in trades_df else np.nan,
        "worst_trade": trades_df['pnl_%'].min() if 'pnl_%' in trades_df else np.nan,
        "median_pnl": trades_df['pnl_%'].median() if 'pnl_%' in trades_df else np.nan,
        "num_wins": (trades_df['pnl_%'] > 0).sum() if 'pnl_%' in trades_df else np.nan,
        "num_losses": (trades_df['pnl_%'] < 0).sum() if 'pnl_%' in trades_df else np.nan,
    }
    return stats

def print_performance_report(equity_curve, trades_df):
    """
    Udskriv en samlet rapport for equity og trades.
    """
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    stats = calculate_trade_stats(trades_df)

    print("===== PERFORMANCE REPORT =====")
    print(f"Sharpe ratio   : {sharpe:.2f}")
    print(f"Sortino ratio  : {sortino:.2f}")
    print(f"Max Drawdown   : {max_dd:.2%}")
    print(f"Win-rate       : {stats['win_rate']:.1f}%")
    print(f"Profit factor  : {stats['profit_factor']:.2f}")
    print(f"Expectancy     : {stats['expectancy']:.2f}%")
    print(f"Antal handler  : {stats['total_trades']}")
    print(f"Bedste trade   : {stats['best_trade']:.2f}%")
    print(f"Værste trade   : {stats['worst_trade']:.2f}%")
    print("==============================")
