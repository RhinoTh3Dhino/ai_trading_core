# bot/paper_trader.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime

# Importér strategier
from strategies.advanced_strategies import (
    ema_crossover_strategy,
    ema_rsi_regime_strategy,
    ema_rsi_adx_strategy,
    voting_ensemble
)

# Importér performance metrics
from utils.performance import print_performance_report

# Importér coins og timeframes fra config
from config.config import COINS, TIMEFRAMES

# -- Parametre (kan importeres fra config.py) --
SL = 0.02
TP = 0.04
START_BALANCE = 10000
FEE = 0.0005

def paper_trade(
    df,
    sl=SL, tp=TP, start_balance=START_BALANCE, fee=FEE,
    JOURNAL_PATH="outputs/paper_trades.csv"
):
    balance = start_balance
    equity = [start_balance]
    position = 0
    entry_price = 0
    trades = []
    n_wins = 0
    n_trades = 0

    for i, row in df.iterrows():
        # ENTRY
        if row['signal'] == 1 and position == 0:
            position = 1
            entry_price = row['close']
            entry_time = row['timestamp'] if 'timestamp' in row else i
            trades.append({
                "time": entry_time, "type": "BUY", "price": entry_price, "balance": balance
            })
            n_trades += 1

        # EXIT (strategi eller SL/TP)
        if position == 1:
            pnl = (row['close'] - entry_price) / entry_price
            if row['signal'] == -1 or pnl <= -sl or pnl >= tp:
                exit_type = "SELL" if row['signal'] == -1 else ("TP" if pnl >= tp else "SL")
                fee_total = balance * fee * 2
                balance = balance * (1 + pnl) - fee_total
                equity.append(balance)
                exit_time = row['timestamp'] if 'timestamp' in row else i
                trades.append({
                    "time": exit_time,
                    "type": exit_type,
                    "price": row['close'],
                    "pnl_%": round(pnl*100, 2),
                    "balance": balance
                })
                if pnl > 0:
                    n_wins += 1
                position = 0
                entry_price = 0
        equity.append(balance)

    # Luk åben position til sidst hvis åben
    if position == 1:
        final_price = df.iloc[-1]['close']
        pnl = (final_price - entry_price) / entry_price
        fee_total = balance * fee * 2
        balance = balance * (1 + pnl) - fee_total
        equity.append(balance)
        trades.append({
            "time": df.iloc[-1].get('timestamp', len(df)-1),
            "type": "FORCE_EXIT",
            "price": final_price,
            "pnl_%": round(pnl*100, 2),
            "balance": balance
        })

    trades_df = pd.DataFrame(trades)
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    trades_df.to_csv(JOURNAL_PATH, index=False)

    # Performance metrics
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
    print(f"Slutbalance: {balance:.2f}")
    print(f"Antal handler: {n_trades}")
    print(f"Win-rate: {win_rate:.1f}%")
    print(f"Journal gemt: {JOURNAL_PATH}")

    # --- Ekstra: Udskriv professionelle performance-metrics ---
    print_performance_report(
        equity_curve=equity,
        trades_df=trades_df
    )

    # --- (Valgfrit) Plot equity curve ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(equity)
        plt.title("Equity Curve")
        plt.xlabel("Step")
        plt.ylabel("Balance")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

    return balance, trades_df

if __name__ == "__main__":
    # Batch-loop over alle coins og timeframes!
    for symbol in COINS:
        for tf in TIMEFRAMES:
            feature_path = f"outputs/feature_data/{symbol.lower()}_{tf}_features_v1.3_{datetime.now().strftime('%Y%m%d')}.csv"
            if not os.path.exists(feature_path):
                print(f"❌ Featurefil mangler: {feature_path}")
                continue
            print(f"\n=== Backtester {symbol} {tf} ===")
            df = pd.read_csv(feature_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Vælg strategi (skift let til ema_rsi_regime_strategy(df) eller voting_ensemble(df))
            df = voting_ensemble(df)

            # Gem versioneret journal pr. batch-run
            journal_path = (
                f"outputs/paper_trades_{symbol.lower()}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            paper_trade(df, JOURNAL_PATH=journal_path)
