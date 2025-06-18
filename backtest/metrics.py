# metrics.py
import pandas as pd
from backtest.backtest import run_backtest, calc_backtest_metrics

def run_and_score(df, signals):
    """Kør backtest og returner standard-metrics dict."""
    trades_df, balance_df = run_backtest(df.copy(), signals=signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    metrics["num_trades"] = len(trades_df)
    return metrics

def evaluate_strategies(
    df,
    ml_signals,
    rsi_signals,
    macd_signals,
    ensemble_signals,
    trades_df=None,
    balance_df=None
):
    """
    Evaluer ML, RSI, MACD og ensemble. Du kan give færdige signal-arrays.
    trades_df og balance_df bruges kun til ensemble, hvis de er medsendt (ellers køres run_and_score).
    """
    ml_metrics = run_and_score(df, ml_signals)
    rsi_metrics = run_and_score(df, rsi_signals)
    macd_metrics = run_and_score(df, macd_signals)
    
    # Brug eksisterende trades_df/balance_df hvis medsendt (dvs. allerede lavet ensemble-backtest i engine)
    if trades_df is not None and balance_df is not None:
        ensemble_metrics = calc_backtest_metrics(trades_df, balance_df)
        ensemble_metrics["num_trades"] = len(trades_df)
    else:
        ensemble_metrics = run_and_score(df, ensemble_signals)
    
    return {
        "ML": ml_metrics,
        "RSI": rsi_metrics,
        "MACD": macd_metrics,
        "ENSEMBLE": ensemble_metrics
    }
