import os
from datetime import datetime

import numpy as np
import pandas as pd

from strategies.advanced_strategies import ema_crossover_strategy
from utils.performance import calculate_performance_metrics
from utils.project_path import PROJECT_ROOT

# strategies/gridsearch_strategies.py


def grid_search_sl_tp_ema(
    df,
    sl_grid=[0.01, 0.02, 0.03],
    tp_grid=[0.02, 0.04, 0.06],
    ema_fast_grid=[9, 13, 21],
    ema_slow_grid=[21, 34, 55],
    regime_only=True,
    regime_col="regime",
    regime_value="bull",
    # AUTO PATH CONVERTED
    log_path=PROJECT_ROOT / "outputs" / "gridsearch/gridsearch_results.csv",
    strategy_func=ema_crossover_strategy,
    rename_to_ema9_21=True,
    top_n=5,
):
    results = []
    for sl in sl_grid:
        for tp in tp_grid:
            for ema_fast in ema_fast_grid:
                for ema_slow in ema_slow_grid:
                    if ema_fast >= ema_slow:
                        continue

                    test_df = df.copy()
                    # Tilføj EMA-kolonner (kun hvis de ikke findes)
                    for ema_len in [ema_fast, ema_slow]:
                        ema_col = f"ema_{ema_len}"
                        if ema_col not in test_df.columns:
                            test_df[ema_col] = (
                                test_df["close"].ewm(span=ema_len, adjust=False).mean()
                            )

                    # Evt. omdøb til ema_9/ema_21
                    if rename_to_ema9_21:
                        rename_map = {
                            f"ema_{ema_fast}": "ema_9",
                            f"ema_{ema_slow}": "ema_21",
                        }
                        strat_df = test_df.rename(columns=rename_map)
                    else:
                        strat_df = test_df

                    # Kør strategi (returnerer df med 'signal'-kolonne)
                    strat_df = strategy_func(strat_df, fast=ema_fast, slow=ema_slow)

                    # Regime-filter (fx kun bull)
                    if regime_only and regime_col in strat_df.columns:
                        strat_df = strat_df[strat_df[regime_col] == regime_value].copy()

                    # Simpel backtest
                    balance, trades_df = paper_trade_simple(strat_df, sl=sl, tp=tp)
                    perf = calculate_performance_metrics(trades_df["balance"], trades_df)
                    perf.update(
                        {
                            "sl": sl,
                            "tp": tp,
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "final_balance": balance,
                        }
                    )
                    results.append(perf)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        results_df = results_df.sort_values("sharpe", ascending=False)
        results_df.to_csv(log_path, index=False)
        print(f"✅ Gemte gridsearch-resultater til {log_path}")
        print(results_df.head(top_n))
    else:
        print("❌ Ingen resultater fra gridsearch (tjek data og strategi).")
    return results_df


def paper_trade_simple(df, sl=0.02, tp=0.04, start_balance=10000, fee=0.0005):
    balance = start_balance
    equity = [start_balance]
    position = 0
    entry_price = 0
    trades = []
    for i, row in df.iterrows():
        signal = row["signal"] if "signal" in row else 0
        price = row["close"]
        ts = row["timestamp"] if "timestamp" in row else i

        if signal == 1 and position == 0:
            position = 1
            entry_price = price
            trades.append({"time": ts, "type": "BUY", "price": entry_price, "balance": balance})
        if position == 1:
            pnl = (price - entry_price) / entry_price
            if signal == -1 or pnl <= -sl or pnl >= tp:
                fee_total = balance * fee * 2
                balance = balance * (1 + pnl) - fee_total
                trades.append(
                    {
                        "time": ts,
                        "type": "SELL",
                        "price": price,
                        "pnl_%": round(pnl * 100, 2),
                        "balance": balance,
                    }
                )
                position = 0
                entry_price = 0
        equity.append(balance)
    # Luk åben til sidst
    if position == 1:
        last_price = df.iloc[-1]["close"]
        ts = df.iloc[-1]["timestamp"] if "timestamp" in df.columns else len(df) - 1
        pnl = (last_price - entry_price) / entry_price
        fee_total = balance * fee * 2
        balance = balance * (1 + pnl) - fee_total
        trades.append(
            {
                "time": ts,
                "type": "FORCE_EXIT",
                "price": last_price,
                "pnl_%": round(pnl * 100, 2),
                "balance": balance,
            }
        )
    trades_df = pd.DataFrame(trades)
    # For metrics (fungerer med calculate_performance_metrics)
    if not trades_df.empty:
        trades_df["balance"] = trades_df["balance"].ffill()
    return balance, trades_df


# --- Udvid nemt med flere gridsearch-funktioner og strategier! ---
