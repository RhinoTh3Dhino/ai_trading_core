import sys, os
import pandas as pd
import numpy as np
import datetime
import subprocess

# Dynamisk projektroot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Utils og robusthed
from utils.file_utils import save_with_metadata
from utils.robust_utils import safe_run

# Ensemble og strategier (let at udvide)
from ensemble.majority_vote_ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

# Versionskontrol fra Git
def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def run_backtest(df, signals=None, initial_balance=1000, fee=0.00075, sl_pct=0.02, tp_pct=0.03):
    df = df.copy()
    # Flexibel timestamp-håndtering
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    for col in ["timestamp", "close"]:
        if col not in df.columns:
            raise ValueError(f"❌ Mangler kolonnen '{col}' i DataFrame til backtest! ({list(df.columns)})")
    if signals is not None:
        df["signal"] = signals
    elif "signal" not in df.columns:
        raise ValueError("Ingen signaler angivet til backtest!")

    trades, balance_log = [], []
    balance = initial_balance
    position, entry_price, entry_time = None, 0, None

    for i, row in df.iterrows():
        price, signal, timestamp = row["close"], row["signal"], row["timestamp"]

        # SL/TP kun for long
        if position == "long":
            change = (price - entry_price) / entry_price
            if change <= -sl_pct:
                balance *= (1 - fee)
                trades.append({"timestamp": timestamp, "type": "SL", "price": price, "balance": balance})
                position = None
            elif change >= tp_pct:
                balance *= (1 - fee)
                trades.append({"timestamp": timestamp, "type": "TP", "price": price, "balance": balance})
                position = None

        if position is None and signal == 1:
            entry_price, entry_time = price, timestamp
            position = "long"
            trades.append({"timestamp": timestamp, "type": "BUY", "price": price, "balance": balance})

        if position == "long" and i == df.index[-1]:
            pct = (price - entry_price) / entry_price
            balance *= (1 + pct - fee)
            trades.append({"timestamp": timestamp, "type": "CLOSE", "price": price, "balance": balance})
            position = None

        balance_log.append({"timestamp": timestamp, "balance": balance})

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_log)
    if "balance" not in balance_df.columns or len(balance_df) == 0:
        print("❌ FEJL: Ingen balance-kolonne i balance_df! Her er head():")
        print(balance_df.head())
    return trades_df, balance_df

def calc_backtest_metrics(trades_df, balance_df, initial_balance=1000):
    if "balance" not in balance_df.columns or len(balance_df) == 0:
        print("❌ FEJL: balance_df tom eller mangler kolonne 'balance'. Kan ikke beregne metrics.")
        return {"profit_pct": 0, "win_rate": 0, "drawdown_pct": 0, "num_trades": 0}

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
    row = {"timestamp": timestamp, "version": version, "git_hash": git_hash, **metrics}
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"✅ Backtest-metrics logget til: {csv_path}")

def main():
    # --- Dynamisk indlæsning af featurefil ---
    feature_path = "outputs/feature_data/btc_1h_features_v_test_20250610.csv"  # Udskift ved behov
    df = pd.read_csv(feature_path)
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    print("Indlæst data med kolonner:", list(df.columns))

    # --- Debug på inputfeatures ---
    for feat in ["rsi_14", "macd", "macd_signal"]:
        if feat in df.columns:
            print(f"{feat} describe():", df[feat].describe())
        else:
            print(f"❌ FEJL: {feat} findes ikke i features!")

    # --- Dummy ML-signaler (udskift med din egen predict-funktion) ---
    np.random.seed(42)
    ml_signals = np.random.choice([1, 0, -1], size=len(df))

    # --- Regelbaserede strategier ---
    rsi_signals = rsi_rule_based_signals(df, low=45, high=55)    # Juster thresholds
    macd_signals = macd_cross_signals(df)

    print("\nSignal distribution ML/RSI/MACD:",
          pd.Series(ml_signals).value_counts().to_dict(),
          pd.Series(rsi_signals).value_counts().to_dict(),
          pd.Series(macd_signals).value_counts().to_dict())

    # --- Ensemble voting ---
    ensemble_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals)
    df["signal"] = ensemble_signals

    # --- Kør backtest ---
    trades_df, balance_df = run_backtest(df)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    save_backtest_results(metrics, version="v1.2.0-ensemble")
    print("Backtest-metrics:", metrics)
    save_with_metadata(balance_df, "data/balance.csv", version="v1.2.0-ensemble")
    save_with_metadata(trades_df, "data/trades.csv", version="v1.2.0-ensemble")

if __name__ == "__main__":
    safe_run(main)
