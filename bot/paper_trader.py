# bot/paper_trader.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

# -- Parametre (kan importeres fra config.py) --
SL = 0.02  # Stop-loss (2%)
TP = 0.04  # Take-profit (4%)
START_BALANCE = 10000
FEE = 0.0005  # Simuler evt. trading fee (0.05%)
FEATURE_PATH = "outputs/feature_data/btcusdt_1h_features_v1.3_20250630.csv"
JOURNAL_PATH = "outputs/paper_trades.csv"

def ema_crossover_strategy(df):
    """Buy: EMA9 > EMA21. Sell: EMA9 < EMA21."""
    df['signal'] = 0
    df.loc[df['ema_9'] > df['ema_21'], 'signal'] = 1
    df.loc[df['ema_9'] < df['ema_21'], 'signal'] = -1
    return df

def paper_trade(df, sl=SL, tp=TP, start_balance=START_BALANCE, fee=FEE):
    balance = start_balance
    equity = [start_balance]
    position = 0  # 0: ingen, 1: long
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
            # Tag profit eller stop-loss
            if row['signal'] == -1 or pnl <= -sl or pnl >= tp:
                exit_type = "SELL" if row['signal'] == -1 else ("TP" if pnl >= tp else "SL")
                fee_total = balance * fee * 2  # Simuler køb+salgsfee
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
    # -- Gem journal --
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    trades_df.to_csv(JOURNAL_PATH, index=False)

    # -- Performance metrics --
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
    print(f"Slutbalance: {balance:.2f}")
    print(f"Antal handler: {n_trades}")
    print(f"Win-rate: {win_rate:.1f}%")
    print(f"Journal gemt: {JOURNAL_PATH}")

    # -- (Valgfrit) Plot equity curve --
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
    # 1. Læs features (direkte fra cleaning-pipeline)
    df = pd.read_csv(FEATURE_PATH)
    # 2. Sørg for at 'timestamp' er datetime (valgfrit)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    # 3. Kør strategi
    df = ema_crossover_strategy(df)
    # 4. Backtest
    paper_trade(df)
