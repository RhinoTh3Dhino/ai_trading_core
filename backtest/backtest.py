from utils.project_path import PROJECT_ROOT
# backtest/backtest.py

import pandas as pd
import numpy as np
import datetime
import subprocess
import argparse


from utils.file_utils import save_with_metadata
from utils.robust_utils import safe_run
from utils.telegram_utils import send_message, send_image
from utils.log_utils import log_device_status

from ensemble.ensemble_predict import ensemble_predict
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals
from strategies.ema_cross_strategy import ema_cross_signals
from strategies.advanced_strategies import (
    ema_crossover_strategy,
    ema_rsi_regime_strategy,
    ema_rsi_adx_strategy,
    rsi_mean_reversion,
    regime_ensemble,
    voting_ensemble,
    add_adaptive_sl_tp,
)
from strategies.gridsearch_strategies import grid_search_sl_tp_ema
from utils.metrics_utils import advanced_performance_metrics

# === Monitoring-parametre fra config ===
try:
    from config.monitoring_config import (
        ALARM_THRESHOLDS,
        ALERT_ON_DRAWNDOWN,
        ALERT_ON_WINRATE,
        ALERT_ON_PROFIT,
        ENABLE_MONITORING,
    )
except ImportError:
    ALARM_THRESHOLDS = {"drawdown": -20, "winrate": 0.20, "profit": -10}
    ALERT_ON_DRAWNDOWN = True
    ALERT_ON_WINRATE = True
    ALERT_ON_PROFIT = True
    ENABLE_MONITORING = True

from utils.monitoring_utils import send_live_metrics

FORCE_DEBUG = False
FORCE_DUMMY_TRADES = False
FEE = 0.0004
SL_PCT = 0.006
TP_PCT = 0.012
ALLOW_SHORT = True

def walk_forward_splits(df, train_size=0.6, test_size=0.2, step_size=0.1, min_train=20):
    n = len(df)
    train_len = int(n * train_size)
    test_len = int(n * test_size)
    step_len = int(n * step_size)
    start = 0
    splits = []
    while start + train_len + test_len <= n:
        train_idx = np.arange(start, start + train_len)
        test_idx = np.arange(start + train_len, start + train_len + test_len)
        if len(train_idx) >= min_train and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
        start += step_len
    return splits

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

def force_trade_signals(length):
    signals = []
    for i in range(length):
        signals.append(1 if i % 2 == 0 else -1)
    return np.array(signals)

def clean_signals(signals, length):
    if isinstance(signals, pd.Series):
        signals = signals.values
    signals = np.array(signals).astype(int)
    if len(signals) != length:
        print(f"[ADVARSEL] Signal-længde ({len(signals)}) matcher ikke data-længde ({length}) – tilpasser med 0.")
        out = np.zeros(length, dtype=int)
        out[:min(len(signals), length)] = signals[:min(len(signals), length)]
        return out
    return signals

def run_backtest(
    df: pd.DataFrame,
    signals: list = None,
    initial_balance: float = 1000,
    fee: float = FEE,
    sl_pct: float = SL_PCT,
    tp_pct: float = TP_PCT
) -> tuple:
    df = df.copy()
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    for col in ["timestamp", "close"]:
        if col not in df.columns:
            raise ValueError(f"❌ Mangler kolonnen '{col}' i DataFrame til backtest! ({list(df.columns)})")
    if signals is not None:
        signals = clean_signals(signals, len(df))
        df["signal"] = signals
    elif "signal" not in df.columns:
        raise ValueError("Ingen signaler angivet til backtest!")
    if "ema_200" in df.columns and df[["close", "ema_200"]].isnull().any().any():
        raise ValueError("❌ DataFrame indeholder NaN i 'close' eller 'ema_200'!")
    df = compute_regime(df)
    trades, balance_log = [], []
    balance = initial_balance
    position, entry_price, entry_time, entry_regime, direction = None, 0, None, None, None

    for i, row in df.iterrows():
        price, signal, timestamp = row["close"], row["signal"], row["timestamp"]
        regime = row["regime"] if "regime" in row else "unknown"

        if FORCE_DEBUG:
            signal = 1 if i % 2 == 0 else -1

        # EXIT-LOGIK FOR AKTIV POSITION
        if position is not None:
            change = (price - entry_price) / entry_price if direction == "long" else (entry_price - price) / entry_price
            if direction == "long":
                if change <= -sl_pct:
                    balance *= (1 - fee)
                    trades.append({"timestamp": timestamp, "type": "SL", "price": price, "balance": balance, "regime": entry_regime, "profit": change, "drawdown": None, "side": "long"})
                    position = None
                elif change >= tp_pct:
                    balance *= (1 - fee)
                    trades.append({"timestamp": timestamp, "type": "TP", "price": price, "balance": balance, "regime": entry_regime, "profit": change, "drawdown": None, "side": "long"})
                    position = None
            elif direction == "short":
                if change <= -sl_pct:
                    balance *= (1 - fee)
                    trades.append({"timestamp": timestamp, "type": "SL", "price": price, "balance": balance, "regime": entry_regime, "profit": change, "drawdown": None, "side": "short"})
                    position = None
                elif change >= tp_pct:
                    balance *= (1 - fee)
                    trades.append({"timestamp": timestamp, "type": "TP", "price": price, "balance": balance, "regime": entry_regime, "profit": change, "drawdown": None, "side": "short"})
                    position = None

        # ENTRY LOGIK
        if position is None:
            if signal == 1:
                entry_price, entry_time = price, timestamp
                entry_regime = regime
                position = "open"
                direction = "long"
                balance *= (1 - fee)
                trades.append({"timestamp": timestamp, "type": "BUY", "price": price, "balance": balance, "regime": regime, "profit": 0, "drawdown": None, "side": "long"})
            elif signal == -1 and ALLOW_SHORT:
                entry_price, entry_time = price, timestamp
                entry_regime = regime
                position = "open"
                direction = "short"
                balance *= (1 - fee)
                trades.append({"timestamp": timestamp, "type": "SELL", "price": price, "balance": balance, "regime": regime, "profit": 0, "drawdown": None, "side": "short"})

        if position is not None and i == df.index[-1]:
            pct = (price - entry_price) / entry_price if direction == "long" else (entry_price - price) / entry_price
            balance *= (1 + pct - fee)
            trades.append({"timestamp": timestamp, "type": "CLOSE", "price": price, "balance": balance, "regime": entry_regime, "profit": pct, "drawdown": None, "side": direction})
            position = None
        balance_log.append({"timestamp": timestamp, "balance": balance, "close": price})

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_log)

    # --- Tilføj drawdown ---
    if not balance_df.empty:
        peak = balance_df["balance"].cummax()
        dd = (balance_df["balance"] - peak) / peak
        balance_df["drawdown"] = dd * 100
        if not trades_df.empty:
            trades_df["drawdown"] = np.interp(
                pd.to_datetime(trades_df["timestamp"]).astype('int64'),
                pd.to_datetime(balance_df["timestamp"]).astype('int64'),
                balance_df["drawdown"].values
            )

    # Sikrer alle kolonner
    for col in ["timestamp", "type", "price", "balance", "regime", "side"]:
        if col not in trades_df.columns:
            trades_df[col] = None
    for col in ["timestamp", "close"]:
        if col not in balance_df.columns:
            balance_df[col] = None

    if FORCE_DUMMY_TRADES and (trades_df.empty or trades_df.shape[0] < 2):
        print("‼️ FORCE DEBUG: Ingen rigtige handler fundet – genererer dummy trades for test!")
        trades_df = pd.DataFrame([
            {"timestamp": df.iloc[0]["timestamp"], "type": "BUY", "price": df.iloc[0]["close"], "balance": initial_balance, "regime": df.iloc[0]["regime"], "profit": 0, "drawdown": None, "side": "long"},
            {"timestamp": df.iloc[-1]["timestamp"], "type": "TP", "price": df.iloc[-1]["close"], "balance": initial_balance * 1.1, "regime": df.iloc[-1]["regime"], "profit": 0.1, "drawdown": None, "side": "long"},
            {"timestamp": df.iloc[-1]["timestamp"], "type": "SL", "price": df.iloc[-1]["close"], "balance": initial_balance * 1.05, "regime": df.iloc[-1]["regime"], "profit": -0.05, "drawdown": None, "side": "long"}
        ])
        balance_df = pd.DataFrame([
            {"timestamp": df.iloc[0]["timestamp"], "balance": initial_balance, "close": df.iloc[0]["close"]},
            {"timestamp": df.iloc[-1]["timestamp"], "balance": initial_balance * 1.1, "close": df.iloc[-1]["close"]},
            {"timestamp": df.iloc[-1]["timestamp"], "balance": initial_balance * 1.05, "close": df.iloc[-1]["close"]}
        ])
    print("TRADES DF:\n", trades_df)
    print("BALANCE DF:\n", balance_df)
    if len(trades_df) < 2:
        print("‼️ ADVARSEL: Få eller ingen handler genereret!")
    return trades_df, balance_df

def calc_backtest_metrics(trades_df, balance_df, initial_balance=1000):
    metrics = advanced_performance_metrics(trades_df, balance_df, initial_balance)
    out = {
        "profit_pct": metrics.get("profit_pct", 0),
        "win_rate": metrics.get("win_rate", 0),
        "drawdown_pct": metrics.get("max_drawdown", 0),
        "num_trades": len(trades_df[trades_df["type"].isin(["TP", "SL", "CLOSE"])]),
        "max_consec_losses": metrics.get("max_consec_losses", 0),
        "recovery_bars": metrics.get("recovery_bars", -1),
        "profit_factor": metrics.get("profit_factor", "N/A"),
        "sharpe": metrics.get("sharpe", 0),
        "sortino": metrics.get("sortino", 0),
    }
    return out

# AUTO PATH CONVERTED
def save_backtest_results(metrics, version="v1", csv_path=PROJECT_ROOT / "data" / "backtest_results.csv"):
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
# AUTO PATH CONVERTED
    parser.add_argument("--feature_path", type=str, default=PROJECT_ROOT / "outputs" / "feature_data/btc_1h_features_v_test_20250610.csv")
# AUTO PATH CONVERTED
    parser.add_argument("--results_path", type=str, default=PROJECT_ROOT / "data" / "backtest_results.csv")
# AUTO PATH CONVERTED
    parser.add_argument("--balance_path", type=str, default=PROJECT_ROOT / "data" / "balance.csv")
# AUTO PATH CONVERTED
    parser.add_argument("--trades_path", type=str, default=PROJECT_ROOT / "data" / "trades.csv")
    parser.add_argument("--strategy", type=str, default="ensemble", choices=["ensemble", "voting", "regime", "ema_rsi", "meanrev"])
    parser.add_argument("--gridsearch", action="store_true")
    parser.add_argument("--voting", type=str, default="majority", choices=["majority", "weighted", "sum"])
    parser.add_argument("--debug_ensemble", action="store_true")
    parser.add_argument("--walkforward", action="store_true", help="Aktiver walk-forward analyse")
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--force_trades", action="store_true", help="Tving signaler til BUY/SELL for test/debug")
    return parser.parse_args()

def load_csv_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if str(first_line).startswith("#"):
        print("[INFO] Meta-header fundet i CSV – loader med skiprows=1")
        return pd.read_csv(file_path, skiprows=1)
    else:
        return pd.read_csv(file_path)

def main():
    args = parse_args()
    log_device_status(
        context="backtest",
        extra={"strategy": args.strategy, "feature_file": args.feature_path},
        telegram_func=send_message
    )

    df = load_csv_auto(args.feature_path)
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    print("Indlæst data med kolonner:", list(df.columns))
    df = compute_regime(df)

    if args.gridsearch:
        print("🔍 Kører gridsearch for EMA-strategier...")
        results_df = grid_search_sl_tp_ema(
            df,
            sl_grid=np.linspace(0.003, 0.012, 5),
            tp_grid=np.linspace(0.008, 0.025, 5),
            ema_fast_grid=[7, 9, 12, 15],
            ema_slow_grid=[21, 30, 34, 55],
            regime_only=False,
            top_n=10,
# AUTO PATH CONVERTED
            log_path=PROJECT_ROOT / "outputs" / "gridsearch/ema_gridsearch_results.csv"
        )
        print(results_df.head(10))
        return

    # === WALK-FORWARD ANALYSE ===
    if args.walkforward:
        splits = walk_forward_splits(df, train_size=args.train_size, test_size=args.test_size, step_size=args.step_size)
        all_metrics = []
        for i, (train_idx, test_idx) in enumerate(splits):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx].copy()
            if args.force_trades:
                signals = np.random.choice([1, -1], size=len(df_test))
            else:
                ml_signals   = rsi_rule_based_signals(df_test, low=35, high=65)
                rsi_signals  = rsi_rule_based_signals(df_test, low=40, high=60)
                macd_signals = macd_cross_signals(df_test)
                ema_signals  = ema_cross_signals(df_test)
                signals = ensemble_predict(
                    ml_signals,
                    rsi_signals,
                    rule_preds=macd_signals,
                    extra_preds=[ema_signals],
                    voting=args.voting,
                    debug=args.debug_ensemble
                )
            sig_dist = pd.Series(signals).value_counts().to_dict()
            print(f"Signal dist vindue {i+1}: {sig_dist}")
            send_message(f"Signal dist vindue {i+1}: {sig_dist}")
            df_test["signal"] = signals
            trades_df, balance_df = run_backtest(df_test, signals=signals)
            metrics = calc_backtest_metrics(trades_df, balance_df)
            all_metrics.append(metrics)
            summary = f"Walk-forward vindue {i+1}:\n" + "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            if trades_df.empty:
                send_message(f"‼️ Ingen handler i vindue {i+1}! Signal dist: {sig_dist}")
            else:
                send_message(summary)
            # === Monitoring/alarmer på test-vindue
            if ENABLE_MONITORING:
                send_live_metrics(
                    trades_df, balance_df,
                    symbol="WALKFORWARD",
                    timeframe=f"win{i+1}",
                    thresholds=ALARM_THRESHOLDS,
                    alert_on_drawdown=ALERT_ON_DRAWNDOWN,
                    alert_on_winrate=ALERT_ON_WINRATE,
                    alert_on_profit=ALERT_ON_PROFIT
                )
        metrics_df = pd.DataFrame(all_metrics)
        try:
            import matplotlib.pyplot as plt
            os.makedirs("outputs", exist_ok=True)
            metrics_df[["profit_pct", "drawdown_pct"]].plot(marker="o")
            plt.title("Walk-forward performance")
            plt.ylabel("Value")
            plt.xlabel("Vindue")
            plt.grid(True)
            plt.tight_layout()
# AUTO PATH CONVERTED
            plt.savefig(PROJECT_ROOT / "outputs" / "walkforward_performance.png")
# AUTO PATH CONVERTED
            send_image(PROJECT_ROOT / "outputs" / "walkforward_performance.png", caption="📈 Walk-forward analyse")
        except Exception as e:
            print(f"❌ Plot-fejl: {e}")
        return

    if FORCE_DEBUG:
        print("‼️ DEBUG: Forcerer skiftevis BUY/SELL på hele datasættet")
        signals = force_trade_signals(len(df))
    else:
        if args.strategy == "ensemble":
            ml_signals   = rsi_rule_based_signals(df, low=35, high=65)
            rsi_signals  = rsi_rule_based_signals(df, low=40, high=60)
            macd_signals = macd_cross_signals(df)
            ema_signals  = ema_cross_signals(df)
            signals = ensemble_predict(
                ml_signals,
                rsi_signals,
                rule_preds=macd_signals,
                extra_preds=[ema_signals],
                voting=args.voting,
                debug=args.debug_ensemble
            )
        elif args.strategy == "voting":
            signals = voting_ensemble(df)['signal']
        elif args.strategy == "regime":
            signals = regime_ensemble(df)['signal']
        elif args.strategy == "ema_rsi":
            signals = ema_rsi_regime_strategy(df)['signal']
        elif args.strategy == "meanrev":
            signals = rsi_mean_reversion(df)['signal']
        else:
            signals = rsi_rule_based_signals(df)
        print("Signal distribution:", pd.Series(signals).value_counts().to_dict())

    filtered_signals = signals
    df["signal"] = filtered_signals

    trades_df, balance_df = run_backtest(df, signals=filtered_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    save_backtest_results(metrics, version="v1.7.0", csv_path=args.results_path)
    print("Backtest-metrics:", metrics)
    save_with_metadata(balance_df, args.balance_path, version="v1.7.0")
    save_with_metadata(trades_df, args.trades_path, version="v1.7.0")
    reg_stats = regime_performance(trades_df)
    print("Regime performance:", reg_stats)
    try:
        send_message(f"📊 Regime-performance:\n{reg_stats}")
        # === Monitoring/alarmer på fuld run
        if ENABLE_MONITORING:
            send_live_metrics(
                trades_df, balance_df,
                symbol="BACKTEST",
                timeframe="all",
                thresholds=ALARM_THRESHOLDS,
                alert_on_drawdown=ALERT_ON_DRAWNDOWN,
                alert_on_winrate=ALERT_ON_WINRATE,
                alert_on_profit=ALERT_ON_PROFIT
            )
    except Exception as e:
        print(f"⚠️ Telegram-fejl: {e}")

if __name__ == "__main__":
    safe_run(main)