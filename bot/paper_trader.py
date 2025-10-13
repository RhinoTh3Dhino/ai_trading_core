import glob
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import COINS, TIMEFRAMES

# Importér strategier (efter sys.path-trick!)
from strategies.advanced_strategies import (  # Bonus: hvis du vil bruge adaptive SL/TP!
    add_adaptive_sl_tp,
    ema_crossover_strategy,
    ema_rsi_adx_strategy,
    ema_rsi_regime_strategy,
    voting_ensemble,
)
from utils.performance import print_performance_report
from utils.project_path import PROJECT_ROOT

# bot/paper_trader.py


# -- Parametre (kan importeres fra config.py) --
SL = 0.02
TP = 0.04
START_BALANCE = 10000
FEE = 0.0005


def find_latest_feature_file(symbol, tf, version="v1.3"):
    """Finder seneste feature-fil med mønster."""
    pattern = str(
        Path(PROJECT_ROOT)
        / "outputs"
        / "feature_data"
        / f"{symbol.lower()}_{tf}_features_{version}_*.csv"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def plot_trades(df, trades_df, journal_path):
    """
    Plot backtest equity, køb/salg, stop loss og take profit på prisgraf.
    Gemmer graf som PNG i samme mappe som journal_path.
    """
    plt.figure(figsize=(14, 7))
    plt.title("Backtest – Signaler & Exits")

    # Pris graf
    plt.plot(df["timestamp"], df["close"], label="Pris")

    # Markér køb, salg, SL, TP
    buys = trades_df[trades_df["type"] == "BUY"]
    sells = trades_df[trades_df["type"] == "SELL"]
    sls = trades_df[trades_df["type"] == "SL"]
    tps = trades_df[trades_df["type"] == "TP"]

    plt.scatter(buys["time"], buys["price"], marker="^", color="green", label="Køb", s=100)
    plt.scatter(sells["time"], sells["price"], marker="v", color="red", label="Sælg", s=100)

    if not sls.empty:
        plt.scatter(sls["time"], sls["price"], marker="x", color="red", label="Stop Loss", s=100)
    if not tps.empty:
        plt.scatter(
            tps["time"],
            tps["price"],
            marker="*",
            color="gold",
            label="Take Profit",
            s=150,
        )

    plt.xlabel("Tid")
    plt.ylabel("Pris")
    plt.legend()
    plt.tight_layout()

    png_path = str(journal_path).replace(".csv", ".png")
    plt.savefig(png_path)
    print(f"✅ Trade-graf gemt som {png_path}")

    plt.show()


def paper_trade(
    df,
    sl=SL,
    tp=TP,
    start_balance=START_BALANCE,
    fee=FEE,
    JOURNAL_PATH=None,
    use_adaptive_sl_tp=False,
):
    balance = start_balance
    equity = [start_balance]
    position = 0
    entry_price = 0
    trades = []
    n_wins = 0
    n_trades = 0
    entry_row = None  # Til adaptiv SL/TP

    if JOURNAL_PATH is None:
        JOURNAL_PATH = Path(PROJECT_ROOT) / "outputs" / "paper_trades.csv"

    for i, row in df.iterrows():
        # ENTRY
        if row["signal"] == 1 and position == 0:
            position = 1
            entry_price = row["close"]
            entry_row = row
            entry_time = row["timestamp"] if "timestamp" in row else i
            trades.append(
                {
                    "time": entry_time,
                    "type": "BUY",
                    "price": entry_price,
                    "balance": balance,
                }
            )
            n_trades += 1

        # EXIT (strategi eller SL/TP)
        if position == 1:
            pnl = (row["close"] - entry_price) / entry_price

            this_sl = entry_row.get("sl_pct", sl) if use_adaptive_sl_tp else sl
            this_tp = entry_row.get("tp_pct", tp) if use_adaptive_sl_tp else tp

            if row["signal"] == -1 or pnl <= -this_sl or pnl >= this_tp:
                exit_type = "SELL" if row["signal"] == -1 else ("TP" if pnl >= this_tp else "SL")
                fee_total = balance * fee * 2
                balance = balance * (1 + pnl) - fee_total
                equity.append(balance)
                exit_time = row["timestamp"] if "timestamp" in row else i
                trades.append(
                    {
                        "time": exit_time,
                        "type": exit_type,
                        "price": row["close"],
                        "pnl_%": round(pnl * 100, 2),
                        "balance": balance,
                    }
                )
                if pnl > 0:
                    n_wins += 1
                position = 0
                entry_price = 0
                entry_row = None
        equity.append(balance)

    if position == 1:
        final_price = df.iloc[-1]["close"]
        pnl = (final_price - entry_price) / entry_price
        this_sl = entry_row.get("sl_pct", sl) if use_adaptive_sl_tp else sl
        this_tp = entry_row.get("tp_pct", tp) if use_adaptive_sl_tp else tp
        fee_total = balance * fee * 2
        balance = balance * (1 + pnl) - fee_total
        equity.append(balance)
        trades.append(
            {
                "time": df.iloc[-1].get("timestamp", len(df) - 1),
                "type": "FORCE_EXIT",
                "price": final_price,
                "pnl_%": round(pnl * 100, 2),
                "balance": balance,
            }
        )

    trades_df = pd.DataFrame(trades)
    os.makedirs(os.path.dirname(str(JOURNAL_PATH)), exist_ok=True)
    trades_df.to_csv(str(JOURNAL_PATH), index=False)

    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
    print(f"Slutbalance: {balance:.2f}")
    print(f"Antal handler: {n_trades}")
    print(f"Win-rate: {win_rate:.1f}%")
    print(f"Journal gemt: {JOURNAL_PATH}")

    print_performance_report(equity_curve=equity, trades_df=trades_df)

    plot_trades(df, trades_df, JOURNAL_PATH)

    return balance, trades_df


if __name__ == "__main__":
    for symbol in COINS:
        for tf in TIMEFRAMES:
            feature_path = find_latest_feature_file(symbol, tf, version="v1.3")
            if not feature_path:
                print(f"❌ Featurefil mangler: {symbol} {tf} version v1.3")
                continue
            print(f"\n=== Backtester {symbol} {tf} med Voting-Ensemble ===")
            df = pd.read_csv(feature_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Uncomment for adaptive SL/TP
            # df = add_adaptive_sl_tp(df)

            # Vælg strategi (kan byttes ud let)
            # df = ema_crossover_strategy(df)
            # df = ema_rsi_regime_strategy(df)
            # df = ema_rsi_adx_strategy(df)
            df = voting_ensemble(df)

            journal_path = (
                Path(PROJECT_ROOT)
                / "outputs"
                / f"paper_trades_{symbol.lower()}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            paper_trade(df, JOURNAL_PATH=journal_path)
