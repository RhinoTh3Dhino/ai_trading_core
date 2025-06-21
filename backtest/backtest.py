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

# Ensemble og strategier
from ensemble.majority_vote_ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

# Telegram (hvis du vil sende resultater/advarsler)
from utils.telegram_utils import send_message

# --- Step 3: Regime-funktioner ---
def compute_regime(df, ema_col="ema_200", price_col="close"):
    """
    Tilføjer en 'regime'-kolonne til df: 'bull', 'bear', 'neutral'
    """
    if "regime" in df.columns:
        return df
    regime = []
    for idx, row in df.iterrows():
        if row[price_col] > row[ema_col]:
            regime.append("bull")
        elif row[price_col] < row[ema_col]:
            regime.append("bear")
        else:
            regime.append("neutral")
    df["regime"] = regime
    return df

def regime_filter(signals, regime_col, active_regimes=["bull"]):
    """
    Returnerer signalet kun hvis regime er i active_regimes, ellers 0.
    """
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

    # --- Step 3: Sørg for at der er regime-kolonne ---
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
            trades.append({"timestamp": timestamp, "type": "BUY", "price": price, "balance": balance, "regime": regime, "profit": 0, "drawdown": None})

        if position == "long" and i == df.index[-1]:
            pct = (price - entry_price) / entry_price
            balance *= (1 + pct - fee)
            trades.append({"timestamp": timestamp, "type": "CLOSE", "price": price, "balance": balance, "regime": entry_regime, "profit": pct, "drawdown": None})
            position = None

        balance_log.append({"timestamp": timestamp, "balance": balance})

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_log)

    # --- Step 3: Tilføj drawdown & evt. telegram-advarsel ---
    if not balance_df.empty:
        peak = balance_df["balance"].cummax()
        dd = (balance_df["balance"] - peak) / peak
        max_drawdown = dd.min() * 100 if not dd.empty else 0
        balance_df["drawdown"] = dd * 100
        if not trades_df.empty:
            trades_df["drawdown"] = np.interp(
                pd.to_datetime(trades_df["timestamp"]).astype(np.int64),
                pd.to_datetime(balance_df["timestamp"]).astype(np.int64),
                balance_df["drawdown"].values
            )
    else:
        max_drawdown = 0

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
    feature_path = "outputs/feature_data/btc_1h_features_v_test_20250610.csv"
    df = pd.read_csv(feature_path)
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    print("Indlæst data med kolonner:", list(df.columns))

    # --- Step 3: Tilføj regime ---
    df = compute_regime(df)

    # --- Dummy ML-signaler (udskift med din egen predict-funktion) ---
    np.random.seed(42)
    ml_signals = np.random.choice([1, 0, -1], size=len(df))

    # --- Regelbaserede strategier ---
    rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
    macd_signals = macd_cross_signals(df)

    print("\nSignal distribution ML/RSI/MACD:",
          pd.Series(ml_signals).value_counts().to_dict(),
          pd.Series(rsi_signals).value_counts().to_dict(),
          pd.Series(macd_signals).value_counts().to_dict())

    # --- Ensemble voting ---
    ensemble_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals)
    df["signal"] = ensemble_signals

    # --- Step 3: Regime-filter (kun bull) ---
    filtered_signals = regime_filter(df["signal"], df["regime"], active_regimes=["bull"])
    df["signal"] = filtered_signals

    # --- Kør backtest ---
    trades_df, balance_df = run_backtest(df, signals=filtered_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    save_backtest_results(metrics, version="v1.3.0-regime")
    print("Backtest-metrics:", metrics)
    save_with_metadata(balance_df, "data/balance.csv", version="v1.3.0-regime")
    save_with_metadata(trades_df, "data/trades.csv", version="v1.3.0-regime")

    # --- Step 3: Regime performance summary ---
    reg_stats = regime_performance(trades_df)
    print("Regime performance:", reg_stats)
    try:
        send_message(f"📊 Regime-performance:\n{reg_stats}")
        # Telegram-advarsler
        for regime, stats in reg_stats.items():
            if stats["win_rate"] is not None and stats["win_rate"] < 0.2 and regime in ("bear", "neutral"):
                send_message(f"⚠️ Win-rate i {regime}-regime er under 20%!")
    except Exception as e:
        print(f"⚠️ Telegram-fejl: {e}")

if __name__ == "__main__":
    safe_run(main)
