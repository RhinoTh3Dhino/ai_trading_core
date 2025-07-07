import sys, os
import pandas as pd
import numpy as np
import datetime
import subprocess
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_utils import save_with_metadata
from utils.robust_utils import safe_run

from ensemble.majority_vote_ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals
from utils.telegram_utils import send_message

def compute_regime(df: pd.DataFrame, ema_col: str = "ema_200", price_col: str = "close") -> pd.DataFrame:
    if "regime" in df.columns:
        return df
    df["regime"] = np.where(
        df[price_col] > df[ema_col], "bull",
        np.where(df[price_col] < df[ema_col], "bear", "neutral")
    )
    return df

def regime_filter(signals, regime_col, active_regimes=["bull"]):
    return [sig if reg in active_regimes else 0 for sig, reg in zip(signals, regime_col)]

def regime_performance(trades_df, regime_col="regime"):
    if regime_col not in trades_df.columns:
        return {}
    grouped = trades_df.groupby(regime_col)
    results = {}
    for name, group in grouped:
        n = len(group)
        win_rate = (group['profit'] > 0).mean() if n > 0 and 'profit' in group.columns else 0
        profit_pct = group['profit'].sum() if 'profit' in group.columns else 0
        drawdown_pct = group['drawdown'].min() if 'drawdown' in group.columns else None
        results[name] = {
            "num_trades": n,
            "win_rate": win_rate,
            "profit_pct": profit_pct,
            "drawdown_pct": drawdown_pct
        }
    return results

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def run_backtest(
    df: pd.DataFrame,
    signals: list = None,
    initial_balance: float = 1000,
    fee: float = 0.00075,
    sl_pct: float = 0.02,
    tp_pct: float = 0.03
) -> tuple:
    df = df.copy()
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    for col in ["timestamp", "close"]:
        if col not in df.columns:
            raise ValueError(f"❌ Mangler kolonnen '{col}' i DataFrame til backtest! ({list(df.columns)})")
    if signals is not None:
        df["signal"] = signals
    elif "signal" not in df.columns:
        raise ValueError("Ingen signaler angivet til backtest!")
    if df[["close", "ema_200"]].isnull().any().any():
        raise ValueError("❌ DataFrame indeholder NaN i 'close' eller 'ema_200'!")
    df = compute_regime(df)
    trades, balance_log = [], []
    balance = initial_balance
    position, entry_price, entry_time, entry_regime = None, 0, None, None

    for i, row in df.iterrows():
        price, signal, timestamp = row["close"], row["signal"], row["timestamp"]
        regime = row["regime"] if "regime" in row else "unknown"
        # SL/TP kun for long
        if position == "long":
            change = (price - entry_price) / entry_price
            if change <= -sl_pct:
                balance *= (1 - fee)
                trades.append({"timestamp": timestamp, "type": "SL", "price": price, "balance": balance, "regime": entry_regime, "profit": change, "drawdown": None})
                position = None
            elif change >= tp_pct:
                balance *= (1 - fee)
                trades.append({"timestamp": timestamp, "type": "TP", "price": price, "balance": balance, "regime": entry_regime, "profit": change, "drawdown": None})
                position = None
        if position is None and signal == 1:
            entry_price, entry_time = price, timestamp
            entry_regime = regime
            position = "long"
            balance *= (1 - fee)
            trades.append({"timestamp": timestamp, "type": "BUY", "price": price, "balance": balance, "regime": regime, "profit": 0, "drawdown": None})
        elif position is None and signal == -1:
            pass
        if position == "long" and i == df.index[-1]:
            pct = (price - entry_price) / entry_price
            balance *= (1 + pct - fee)
            trades.append({"timestamp": timestamp, "type": "CLOSE", "price": price, "balance": balance, "regime": entry_regime, "profit": pct, "drawdown": None})
            position = None
        balance_log.append({"timestamp": timestamp, "balance": balance, "close": price})

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_log)

    # --- Tilføj drawdown & evt. telegram-advarsel ---
    if not balance_df.empty:
        peak = balance_df["balance"].cummax()
        dd = (balance_df["balance"] - peak) / peak
        max_drawdown = dd.min() * 100 if not dd.empty else 0
        balance_df["drawdown"] = dd * 100
        if not trades_df.empty:
            trades_df["drawdown"] = np.interp(
                pd.to_datetime(trades_df["timestamp"]).astype('int64'),
                pd.to_datetime(balance_df["timestamp"]).astype('int64'),
                balance_df["drawdown"].values
            )
    else:
        max_drawdown = 0

    # --- Sikrer at både trades_df og balance_df har alle test-krævede kolonner ---
    if not trades_df.empty and "close" not in trades_df.columns:
        trades_df["close"] = trades_df["price"]  # Tilføj close som kopi af price, hvis ikke allerede
    for col in ["timestamp", "type", "price", "balance", "regime"]:
        if col not in trades_df.columns:
            trades_df[col] = None
    for col in ["timestamp", "close"]:
        if col not in balance_df.columns:
            balance_df[col] = None

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="outputs/feature_data/btc_1h_features_v_test_20250610.csv")
    parser.add_argument("--results_path", type=str, default="data/backtest_results.csv")
    parser.add_argument("--balance_path", type=str, default="data/balance.csv")
    parser.add_argument("--trades_path", type=str, default="data/trades.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.feature_path)
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    print("Indlæst data med kolonner:", list(df.columns))
    df = compute_regime(df)
    np.random.seed(42)
    ml_signals = np.random.choice([1, 0, -1], size=len(df))
    rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
    macd_signals = macd_cross_signals(df)
    print("\nSignal distribution ML/RSI/MACD:",
          pd.Series(ml_signals).value_counts().to_dict(),
          pd.Series(rsi_signals).value_counts().to_dict(),
          pd.Series(macd_signals).value_counts().to_dict())
    ensemble_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals)
    df["signal"] = ensemble_signals
    filtered_signals = regime_filter(df["signal"], df["regime"], active_regimes=["bull"])
    df["signal"] = filtered_signals
    trades_df, balance_df = run_backtest(df, signals=filtered_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    save_backtest_results(metrics, version="v1.3.0-regime", csv_path=args.results_path)
    print("Backtest-metrics:", metrics)
    save_with_metadata(balance_df, args.balance_path, version="v1.3.0-regime")
    save_with_metadata(trades_df, args.trades_path, version="v1.3.0-regime")
    reg_stats = regime_performance(trades_df)
    print("Regime performance:", reg_stats)
    try:
        send_message(f"📊 Regime-performance:\n{reg_stats}")
        for regime, stats in reg_stats.items():
            if stats["win_rate"] is not None and stats["win_rate"] < 0.2 and regime in ("bear", "neutral"):
                send_message(f"⚠️ Win-rate i {regime}-regime er under 20%!")
    except Exception as e:
        print(f"⚠️ Telegram-fejl: {e}")

if __name__ == "__main__":
    safe_run(main)
