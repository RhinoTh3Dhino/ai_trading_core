# utils/performance.py
import numpy as np
import pandas as pd


def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    sharpe = excess.mean() / (returns.std() + 1e-9) * np.sqrt(periods_per_year)
    return sharpe if np.isfinite(sharpe) else 0.0


def calculate_sortino_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    returns = pd.Series(equity_curve).pct_change().dropna()
    neg_returns = returns[returns < 0]
    downside = neg_returns.std()
    if downside == 0 or len(returns) < 2:
        return 0.0
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    sortino = excess / (downside + 1e-9) * np.sqrt(periods_per_year)
    return sortino if np.isfinite(sortino) else 0.0


def calculate_max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    if len(equity) == 0:
        return 0.0
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / (roll_max + 1e-9)
    return drawdown.min() if len(drawdown) > 0 else 0.0


def calculate_volatility(equity_curve, periods_per_year=252):
    returns = pd.Series(equity_curve).pct_change().dropna()
    volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    return volatility if np.isfinite(volatility) else 0.0


def calculate_calmar_ratio(equity_curve, periods_per_year=252):
    returns = pd.Series(equity_curve).pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    annual_return = (1 + returns.mean()) ** periods_per_year - 1
    max_dd = abs(calculate_max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    calmar = annual_return / max_dd
    return calmar if np.isfinite(calmar) else 0.0


def calculate_winrate(trades_df):
    if "pnl_%" not in trades_df or len(trades_df) == 0:
        return 0.0
    wins = trades_df["pnl_%"] > 0
    return wins.sum() / len(trades_df) * 100


def calculate_profit_factor(trades_df):
    if "pnl_%" not in trades_df or len(trades_df) == 0:
        return 0.0
    profits = trades_df[trades_df["pnl_%"] > 0]["pnl_%"].sum()
    losses = -trades_df[trades_df["pnl_%"] < 0]["pnl_%"].sum()
    pf = profits / losses if losses > 0 else 0.0
    return pf if np.isfinite(pf) else 0.0


def calculate_expectancy(trades_df):
    if "pnl_%" not in trades_df or len(trades_df) == 0:
        return 0.0
    exp = trades_df["pnl_%"].mean()
    return exp if np.isfinite(exp) else 0.0


def calculate_kelly_criterion(trades_df):
    if "pnl_%" not in trades_df or len(trades_df) == 0:
        return 0.0
    win_rate = calculate_winrate(trades_df) / 100
    loss_rate = 1 - win_rate
    avg_win = trades_df[trades_df["pnl_%"] > 0]["pnl_%"].mean()
    avg_loss = abs(trades_df[trades_df["pnl_%"] < 0]["pnl_%"].mean())
    if avg_loss == 0 or np.isnan(avg_win) or np.isnan(avg_loss):
        return 0.0
    b = avg_win / avg_loss
    kelly = win_rate - (loss_rate / b) if b != 0 else 0.0
    return kelly if np.isfinite(kelly) else 0.0


def calculate_profit(equity_curve):
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
    pct_profit = (end / start - 1) * 100 if start != 0 else 0.0
    return abs_profit, pct_profit


def calculate_rolling_sharpe(equity_curve, window=50):
    returns = pd.Series(equity_curve).pct_change().dropna()
    if len(returns) < window:
        return 0.0
    rolling_sharpe = (
        returns.rolling(window).mean() / (returns.rolling(window).std() + 1e-9) * np.sqrt(252)
    )
    return rolling_sharpe.iloc[-1] if not rolling_sharpe.empty else 0.0


def calculate_trade_duration(trades_df):
    if "entry_time" in trades_df and "exit_time" in trades_df and len(trades_df) > 0:
        entry = pd.to_datetime(trades_df["entry_time"])
        exit = pd.to_datetime(trades_df["exit_time"])
        durations = (exit - entry).dt.total_seconds() / 3600  # Timer
        return {
            "mean_trade_duration": durations.mean() if not durations.empty else 0.0,
            "median_trade_duration": durations.median() if not durations.empty else 0.0,
            "max_trade_duration": durations.max() if not durations.empty else 0.0,
        }
    else:
        return {
            "mean_trade_duration": 0.0,
            "median_trade_duration": 0.0,
            "max_trade_duration": 0.0,
        }


def calculate_regime_drawdown(trades_df):
    # Forudsætter en 'regime'-kolonne i trades_df (fx "bull", "bear", "neutral") + balance
    if "regime" not in trades_df or "balance" not in trades_df:
        return {}
    regime_stats = {}
    for regime, group in trades_df.groupby("regime"):
        dd = calculate_max_drawdown(group["balance"])
        regime_stats[f"drawdown_{regime}"] = dd
    return regime_stats


def calculate_trade_stats(trades_df):
    stats = {
        "total_trades": len(trades_df),
        "win_rate": calculate_winrate(trades_df),
        "expectancy": calculate_expectancy(trades_df),
        "profit_factor": calculate_profit_factor(trades_df),
        "kelly_criterion": calculate_kelly_criterion(trades_df),
        "avg_pnl": (
            trades_df["pnl_%"].mean() if "pnl_%" in trades_df and len(trades_df) > 0 else 0.0
        ),
        "best_trade": (
            trades_df["pnl_%"].max() if "pnl_%" in trades_df and len(trades_df) > 0 else 0.0
        ),
        "worst_trade": (
            trades_df["pnl_%"].min() if "pnl_%" in trades_df and len(trades_df) > 0 else 0.0
        ),
        "median_pnl": (
            trades_df["pnl_%"].median() if "pnl_%" in trades_df and len(trades_df) > 0 else 0.0
        ),
        "num_wins": (
            (trades_df["pnl_%"] > 0).sum() if "pnl_%" in trades_df and len(trades_df) > 0 else 0.0
        ),
        "num_losses": (
            (trades_df["pnl_%"] < 0).sum() if "pnl_%" in trades_df and len(trades_df) > 0 else 0.0
        ),
    }
    # Trade duration (mean/median/max)
    durations = calculate_trade_duration(trades_df)
    stats.update(durations)
    # NB: Vi returnerer aldrig dicts i stats!
    for k, v in stats.items():
        if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
            stats[k] = 0.0
    return stats


def print_performance_report(equity_curve, trades_df):
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    volatility = calculate_volatility(equity_curve)
    calmar = calculate_calmar_ratio(equity_curve)
    abs_profit, pct_profit = calculate_profit(equity_curve)
    stats = calculate_trade_stats(trades_df)
    rolling_sharpe = calculate_rolling_sharpe(equity_curve)
    regime_dd = calculate_regime_drawdown(trades_df)

    print("===== PERFORMANCE REPORT =====")
    print(f"Sharpe ratio    : {sharpe:.2f}")
    print(f"Sortino ratio   : {sortino:.2f}")
    print(f"Calmar ratio    : {calmar:.2f}")
    print(f"Volatilitet     : {volatility:.2f}")
    print(f"Rolling Sharpe  : {rolling_sharpe:.2f}")
    print(f"Max Drawdown    : {max_dd:.2%}")
    print(f"Profit          : {abs_profit:.2f} ({pct_profit:.2f}%)")
    print(f"Win-rate        : {stats['win_rate']:.1f}%")
    print(f"Profit factor   : {stats['profit_factor']:.2f}")
    print(f"Kelly Criterion : {stats['kelly_criterion']:.2f}")
    print(f"Expectancy      : {stats['expectancy']:.2f}%")
    print(f"Antal handler   : {stats['total_trades']}")
    print(f"Gns. varighed   : {stats.get('mean_trade_duration', 0):.2f} timer")
    print(f"Bedste trade    : {stats['best_trade']:.2f}%")
    print(f"Værste trade    : {stats['worst_trade']:.2f}%")
    for regime, dd in regime_dd.items():
        print(f"Drawdown ({regime}): {dd:.2%}")
    print("==============================")


def calculate_performance_metrics(equity_curve, trades_df):
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    volatility = calculate_volatility(equity_curve)
    calmar = calculate_calmar_ratio(equity_curve)
    abs_profit, pct_profit = calculate_profit(equity_curve)
    stats = calculate_trade_stats(trades_df)
    rolling_sharpe = calculate_rolling_sharpe(equity_curve)
    regime_dd = calculate_regime_drawdown(trades_df)
    # Udflad regime_dd dict til talfelter
    flat_regime_dd = {}
    if isinstance(regime_dd, dict):
        for k, v in regime_dd.items():
            flat_regime_dd[k] = v if np.isfinite(v) else 0.0

    if hasattr(equity_curve, "iloc"):
        final_balance = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0.0
    elif hasattr(equity_curve, "__getitem__"):
        final_balance = equity_curve[-1] if len(equity_curve) > 0 else 0.0
    else:
        final_balance = 0.0

    # Robusthed mod nan/inf
    def safe_val(x):
        if isinstance(x, float) and (np.isnan(x) or not np.isfinite(x)):
            return 0.0
        return x

    out = {
        "sharpe": safe_val(sharpe),
        "sortino": safe_val(sortino),
        "calmar": safe_val(calmar),
        "volatility": safe_val(volatility),
        "rolling_sharpe": safe_val(rolling_sharpe),
        "max_drawdown": safe_val(max_dd),
        "final_balance": safe_val(final_balance),
        "abs_profit": safe_val(abs_profit),
        "pct_profit": safe_val(pct_profit),
        "win_rate": safe_val(stats["win_rate"]),
        "profit_factor": safe_val(stats["profit_factor"]),
        "kelly_criterion": safe_val(stats["kelly_criterion"]),
        "expectancy": safe_val(stats["expectancy"]),
        "total_trades": safe_val(stats["total_trades"]),
        "best_trade": safe_val(stats["best_trade"]),
        "worst_trade": safe_val(stats["worst_trade"]),
        "mean_trade_duration": safe_val(stats.get("mean_trade_duration", 0)),
        "median_trade_duration": safe_val(stats.get("median_trade_duration", 0)),
        "max_trade_duration": safe_val(stats.get("max_trade_duration", 0)),
    }
    # Tilføj ALDRIG dicts i out! Kun floats/tal.
    out.update(flat_regime_dd)
    return out
