import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import datetime
import subprocess
from utils.file_utils import save_with_metadata
from utils.robust_utils import safe_run  # ← Tilføjet robusthed!

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except:
        return "unknown"

def run_backtest(df, signals=None, initial_balance=1000, fee=0.00075, sl_pct=0.02, tp_pct=0.03):
    df = df.copy()
    # Automatisk støtte for både 'timestamp' og 'datetime'
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    for col in ["timestamp", "close"]:
        if col not in df.columns:
            raise ValueError(f"❌ Mangler kolonnen '{col}' i DataFrame til backtest! Findes disse kolonner? {list(df.columns)}")

    if signals is not None:
        df["signal"] = signals
    elif "signal" not in df.columns:
        raise ValueError("Ingen signaler angivet til backtest!")

    trades = []
    balance_log = []
    balance = initial_balance
    position = None
    entry_price = 0
    entry_time = None

    for i, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]
        timestamp = row["timestamp"]

        # Check åben position for SL/TP (kun long)
        if position == "long":
            change = (price - entry_price) / entry_price
            if change <= -sl_pct:
                balance = balance * (1 - fee)
                trades.append({
                    "timestamp": timestamp,
                    "type": "SL",
                    "price": price,
                    "balance": balance
                })
                position = None
            elif change >= tp_pct:
                balance = balance * (1 - fee)
                trades.append({
                    "timestamp": timestamp,
                    "type": "TP",
                    "price": price,
                    "balance": balance
                })
                position = None

        if position is None:
            if signal == 1:
                entry_price = price
                entry_time = timestamp
                position = "long"
                trades.append({
                    "timestamp": timestamp,
                    "type": "BUY",
                    "price": price,
                    "balance": balance
                })
            elif signal == -1:
                pass  # Placeholder til short

        # Luk position ved sidste bar
        if position == "long" and i == df.index[-1]:
            pct = (price - entry_price) / entry_price
            balance = balance * (1 + pct - fee)
            trades.append({
                "timestamp": timestamp,
                "type": "CLOSE",
                "price": price,
                "balance": balance
            })
            position = None

        balance_log.append({"timestamp": timestamp, "balance": balance})

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_log)
    return trades_df, balance_df

def calc_backtest_metrics(trades_df, balance_df, initial_balance=1000):
    profit = (balance_df["balance"].iloc[-1] - initial_balance) / initial_balance * 100
    trade_types = trades_df["type"].values
    tp_count = np.sum(trade_types == "TP")
    sl_count = np.sum(trade_types == "SL")
    win_rate = tp_count / (tp_count + sl_count) if (tp_count + sl_count) > 0 else 0
    num_trades = len(trades_df[trades_df["type"].isin(["TP", "SL", "CLOSE"])])
    peak = balance_df["balance"].cummax()
    drawdown = (balance_df["balance"] - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    return {
        "profit_pct": round(profit, 2),
        "win_rate": round(win_rate, 4),
        "drawdown_pct": round(max_drawdown, 2),
        "num_trades": int(num_trades)
    }

def save_backtest_results(metrics, version="v1", csv_path="data/backtest_results.csv"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()
    row = {
        "timestamp": timestamp,
        "version": version,
        "git_hash": git_hash,
        **metrics
    }
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"✅ Backtest-metrics logget til: {csv_path}")

def main():
    # Eksempel: Brug din featurefil med signal-kolonne
    df = pd.read_csv("data/BTCUSDT_1h_features.csv")
    # Automatisk kolonne-mapping:
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    if "signal" not in df.columns:
        import numpy as np
        df["signal"] = np.random.choice([1, 0, -1], size=len(df))
    trades_df, balance_df = run_backtest(df)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    save_backtest_results(metrics, version="v1.0.1")
    print("Backtest-metrics:", metrics)
    # Gem resultater med metadata
    save_with_metadata(balance_df, "data/balance.csv", version="v1.0.1")
    save_with_metadata(trades_df, "data/trades.csv", version="v1.0.1")

if __name__ == "__main__":
    safe_run(main)
