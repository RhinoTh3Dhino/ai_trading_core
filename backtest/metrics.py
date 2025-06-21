# metrics.py
import pandas as pd
import numpy as np
from backtest.backtest import run_backtest, calc_backtest_metrics

def run_and_score(df, signals):
    """Kør backtest og returner standard-metrics dict."""
    trades_df, balance_df = run_backtest(df.copy(), signals=signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    metrics["num_trades"] = len(trades_df)
    return metrics

def regime_performance(trades_df, regime_col="regime"):
    """
    Returnerer dict med win-rate, profit, drawdown pr. regime (bull/bear/neutral)
    trades_df skal indeholde kolonnen 'regime'
    Håndterer både tekst og numeriske regime-værdier.
    """
    regime_map = {0: "bull", 1: "bear", 2: "neutral"}
    if regime_col not in trades_df.columns:
        print(f"❌ Regime-kolonne '{regime_col}' ikke fundet i trades_df! Kan ikke lave regime-analyse.")
        return {}
    # Håndter evt. numeriske regime-værdier efter merge
    if trades_df[regime_col].dropna().apply(lambda x: isinstance(x, (int, float))).any():
        trades_df[regime_col] = trades_df[regime_col].map(regime_map).fillna(trades_df[regime_col])
    if trades_df[regime_col].dropna().empty:
        print(f"❌ Ingen regime-values i trades_df! Ingen regime-analyse mulig.")
        return {}
    stats = {}
    for regime in trades_df[regime_col].dropna().unique():
        sub = trades_df[trades_df[regime_col] == regime]
        trade_types = sub["type"].values
        tp_count = np.sum(trade_types == "TP")
        sl_count = np.sum(trade_types == "SL")
        win_rate = tp_count / (tp_count + sl_count) if (tp_count + sl_count) > 0 else 0
        num_trades = len(sub)
        profit_pct = (sub["balance"].iloc[-1] - sub["balance"].iloc[0]) / sub["balance"].iloc[0] * 100 if num_trades > 1 else 0
        stats[str(regime)] = {
            "num_trades": num_trades,
            "win_rate": round(win_rate, 4),
            "profit_pct": round(profit_pct, 2)
        }
    return stats

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
    Evaluer ML, RSI, MACD og ensemble. Robust regime-analyse via asof-merge.
    Nu beskyttet mod alle kendte fejl og merge-problemer.
    """
    ml_metrics = run_and_score(df, ml_signals)
    rsi_metrics = run_and_score(df, rsi_signals)
    macd_metrics = run_and_score(df, macd_signals)

    # Ensemble metrics (brug eksisterende hvis muligt)
    if trades_df is not None and balance_df is not None:
        ensemble_metrics = calc_backtest_metrics(trades_df, balance_df)
        ensemble_metrics["num_trades"] = len(trades_df)
    else:
        ensemble_metrics = run_and_score(df, ensemble_signals)

    # --- Robust regime-analyse for ensemble ---
    try:
        do_regime = (
            trades_df is not None
            and "timestamp" in trades_df.columns
            and "regime" in df.columns
            and not df["regime"].isnull().all()
        )
        if do_regime:
            trades_df_regime = trades_df.copy()
            trades_df_regime["timestamp"] = pd.to_datetime(trades_df_regime["timestamp"], errors="coerce")
            regime_lookup = df[["timestamp", "regime"]].copy()
            regime_lookup["timestamp"] = pd.to_datetime(regime_lookup["timestamp"], errors="coerce")

            # merge_asof matcher nærmeste timestamp med tolerance
            trades_df_regime = pd.merge_asof(
                trades_df_regime.sort_values("timestamp"),
                regime_lookup.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta('1h'),
                suffixes=('', '_feat')
            )
            # Håndter evt. flere regime-kolonner (fx regime_feat efter merge)
            if "regime_feat" in trades_df_regime.columns:
                trades_df_regime["regime"] = trades_df_regime["regime_feat"]
                trades_df_regime.drop(columns=["regime_feat"], inplace=True)
            n_na = trades_df_regime["regime"].isna().sum()
            if n_na > 0:
                print(f"⚠️ Regime-merge: {n_na} handler havde ikke match og sættes til 'ukendt'.")
                trades_df_regime["regime"].fillna("ukendt", inplace=True)
            if "regime" not in trades_df_regime.columns or trades_df_regime["regime"].dropna().empty:
                print("⚠️ Efter merge stadig ingen regime – ingen regime-analyse.")
                ensemble_metrics["regime_stats"] = {}
            else:
                regime_stats = regime_performance(trades_df_regime)
                ensemble_metrics["regime_stats"] = regime_stats
        else:
            print("⚠️ trades_df mangler 'timestamp' eller df mangler 'regime' – springer regime-analyse over.")
            ensemble_metrics["regime_stats"] = {}
    except Exception as e:
        print(f"⚠️ Fejl under regime-analyse: {e}")
        if trades_df is not None:
            print(trades_df.head())
        print(df[["timestamp", "regime"]].head())
        ensemble_metrics["regime_stats"] = {}

    return {
        "ML": ml_metrics,
        "RSI": rsi_metrics,
        "MACD": macd_metrics,
        "ENSEMBLE": ensemble_metrics
    }

# Eksempel på brug i engine.py:
# strat_scores = evaluate_strategies(df, ml_signals, rsi_signals, macd_signals, ensemble_signals, trades_df, balance_df)
# print("Strategi-score pr. regime:", strat_scores["ENSEMBLE"].get("regime_stats", {}))
