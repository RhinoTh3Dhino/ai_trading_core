# strategies/gridsearch_strategies.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.advanced_strategies import ema_crossover_strategy
from utils.performance import calculate_performance_metrics

def grid_search_sl_tp_ema(
    df,
    sl_grid=[0.01, 0.02, 0.03],
    tp_grid=[0.02, 0.04, 0.06],
    ema_fast_grid=[9, 13, 21],
    ema_slow_grid=[21, 34, 55],
    regime_only=True,
    regime_col="regime",
    regime_value=1,
    log_path="outputs/gridsearch/gridsearch_results.csv",
    strategy_func=ema_crossover_strategy,
    top_n=5
):
    results = []

    for sl in sl_grid:
        for tp in tp_grid:
            for ema_fast in ema_fast_grid:
                for ema_slow in ema_slow_grid:
                    if ema_fast >= ema_slow:
                        continue

                    test_df = df.copy()
                    # Skriv EMA-kolonner så strategien altid får ema_9 og ema_21
                    test_df["ema_9"] = test_df["close"].ewm(span=ema_fast, adjust=False).mean()
                    test_df["ema_21"] = test_df["close"].ewm(span=ema_slow, adjust=False).mean()

                    # Kør strategi direkte – INGEN kolonne-rename!
                    strat_df = strategy_func(test_df)

                    # Regime filter (fx kun bull)
                    if regime_only and regime_col in strat_df.columns:
                        strat_df = strat_df[strat_df[regime_col] == regime_value].copy()

                    # Simpel backtest
                    balance, trades_df = paper_trade_simple(strat_df, sl=sl, tp=tp)
                    perf = calculate_performance_metrics(trades_df["balance"], trades_df)
                    perf.update({
                        "sl": sl,
                        "tp": tp,
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "final_balance": balance
                    })
                    results.append(perf)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        results_df.sort_values("sharpe", ascending=False).to_csv(log_path, index=False)
        print(f"✅ Gemte gridsearch-resultater til {log_path}")
        print(results_df.sort_values('sharpe', ascending=False).head(top_n))
    else:
        print("❌ Ingen resultater fra gridsearch (tjek data).")
    return results_df

def paper_trade_simple(df, sl=0.02, tp=0.04, start_balance=10000, fee=0.0005):
    balance = start_balance
    equity = [start_balance]
    position = 0
    entry_price = 0
    trades = []
    for i, row in df.iterrows():
        if row.get('signal', 0) == 1 and position == 0:
            position = 1
            entry_price = row['close']
            entry_time = row['timestamp'] if 'timestamp' in row else i
            trades.append({
                "time": entry_time, "type": "BUY", "price": entry_price, "balance": balance
            })
        if position == 1:
            pnl = (row['close'] - entry_price) / entry_price
            if row.get('signal', 0) == -1 or pnl <= -sl or pnl >= tp:
                fee_total = balance * fee * 2
                balance = balance * (1 + pnl) - fee_total
                equity.append(balance)
                exit_time = row['timestamp'] if 'timestamp' in row else i
                trades.append({
                    "time": exit_time,
                    "type": "SELL",
                    "price": row['close'],
                    "pnl_%": round(pnl*100, 2),
                    "balance": balance
                })
                position = 0
                entry_price = 0
        equity.append(balance)
    # Luk åben til sidst
    if position == 1:
        pnl = (df.iloc[-1]['close'] - entry_price) / entry_price
        fee_total = balance * fee * 2
        balance = balance * (1 + pnl) - fee_total
        equity.append(balance)
        trades.append({
            "time": df.iloc[-1].get('timestamp', len(df)-1),
            "type": "FORCE_EXIT",
            "price": df.iloc[-1]['close'],
            "pnl_%": round(pnl*100, 2),
            "balance": balance
        })
    trades_df = pd.DataFrame(trades)
    return balance, trades_df

# --- Klar til flere gridsearch-funktioner og strategier!
