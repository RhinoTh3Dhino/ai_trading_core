# tests/test_walkforward.py

import sys
import os
import glob
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config.config import COINS, TIMEFRAMES
from strategies.advanced_strategies import ema_crossover_strategy, ema_rsi_regime_strategy, voting_ensemble
from bot.paper_trader import paper_trade as paper_trade_advanced
from strategies.gridsearch_strategies import paper_trade_simple
from utils.performance import calculate_performance_metrics

# --- Telegram integration ---
from utils.telegram_utils import send_image, send_document

DEFAULT_WINDOW_SIZE = 200
MIN_WINDOW_SIZE = 100
STEP_SIZE = 50
TRAIN_SIZE = 0.7

OUTPUT_DIR = "outputs/walkforward/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_feature_file(symbol, timeframe, feature_dir="outputs/feature_data"):
    pattern = f"{feature_dir}/{symbol.lower()}_{timeframe}_features_v1.3_*.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def walkforward_split(df, window_size, train_size, step_size):
    n = len(df)
    if n < window_size:
        if n >= MIN_WINDOW_SIZE:
            window_size = n
        else:
            return
    train_len = int(window_size * train_size)
    test_len = window_size - train_len
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        train_df = df.iloc[start : start + train_len]
        test_df = df.iloc[start + train_len : end]
        yield train_df, test_df, start, end

def plot_walkforward_results(results_df, symbol, tf):
    windows = range(len(results_df))
    plt.figure(figsize=(14,6))
    plt.plot(windows, results_df['test_sharpe'], label='Test Sharpe')
    plt.plot(windows, results_df['test_win_rate'], label='Test Win-rate (%)')
    plt.plot(windows, results_df['test_final_balance'], label='Test Balance')
    plt.plot(windows, results_df['test_pct_profit'], label='Test Profit %')
    plt.plot(windows, results_df['test_max_drawdown'], label='Test Max Drawdown')
    # Bonus-metrics
    if 'test_volatility' in results_df:
        plt.plot(windows, results_df['test_volatility'], label='Test Volatility')
    if 'test_calmar' in results_df:
        plt.plot(windows, results_df['test_calmar'], label='Test Calmar')
    if 'test_buyhold_pct' in results_df:
        plt.plot(windows, results_df['test_buyhold_pct'], label='Buy & Hold (test)')
    plt.title(f"Walkforward Performance for {symbol} {tf}")
    plt.xlabel("Walkforward Window")
    plt.legend()
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}walkforward_plot_{symbol}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.show()
    print(f"✅ Walkforward performance-graf gemt som {filename}")
    return filename

if __name__ == "__main__":
    all_results = []
    splits_count = {}

    STRATEGY = voting_ensemble
    PAPER_TRADE_FUNC = paper_trade_advanced

    for symbol in COINS:
        for tf in TIMEFRAMES:
            feature_path = get_latest_feature_file(symbol, tf)
            if not feature_path:
                print(f"❌ Feature-fil mangler for {symbol} {tf} i outputs/feature_data/")
                continue

            print(f"\n=== Walkforward Validation: {symbol} {tf} ===")
            df = pd.read_csv(feature_path)
            print(f"🔎 {symbol} {tf} har {len(df)} rækker i datasættet.")

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            n = len(df)
            splits = 0
            window_size = min(DEFAULT_WINDOW_SIZE, n) if n >= MIN_WINDOW_SIZE else 0
            if window_size < MIN_WINDOW_SIZE:
                print(f"⚠️ Ikke nok data ({n} rækker) til walkforward på {symbol} {tf}. Skipper...")
                continue

            for train_df, test_df, start, end in walkforward_split(df, window_size, TRAIN_SIZE, STEP_SIZE):
                splits += 1
                try:
                    train_df = STRATEGY(train_df)
                    test_df = STRATEGY(test_df)

                    train_balance, train_trades = PAPER_TRADE_FUNC(train_df)
                    test_balance, test_trades = PAPER_TRADE_FUNC(test_df)

                    if len(train_trades) == 0 or len(test_trades) == 0:
                        print(f"⚠️ Split {start}-{end} har ingen handler. Skipper split.")
                        continue

                    train_metrics = calculate_performance_metrics(train_trades["balance"], train_trades)
                    test_metrics = calculate_performance_metrics(test_trades["balance"], test_trades)

                    if not isinstance(train_metrics, dict) or not isinstance(test_metrics, dict):
                        print(f"⚠️ Metrics returneret ikke dict. Skipper split {start}-{end}.")
                        continue

                    train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

                    # --- Buy & Hold for både train og test ---
                    train_buyhold = np.nan
                    test_buyhold = np.nan
                    if "close" in train_df.columns and "close" in test_df.columns:
                        bh_train_start = train_df["close"].iloc[0]
                        bh_train_end = train_df["close"].iloc[-1]
                        train_buyhold = (bh_train_end / bh_train_start - 1) * 100 if bh_train_start != 0 else np.nan
                        bh_test_start = test_df["close"].iloc[0]
                        bh_test_end = test_df["close"].iloc[-1]
                        test_buyhold = (bh_test_end / bh_test_start - 1) * 100 if bh_test_start != 0 else np.nan

                    result = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "window_start": start,
                        "window_end": end,
                        "strategy": STRATEGY.__name__,
                        "window_size": window_size,
                        "train_buyhold_pct": train_buyhold,
                        "test_buyhold_pct": test_buyhold,
                    }
                    result.update(train_metrics)
                    result.update(test_metrics)
                    all_results.append(result)

                    print(f"[{symbol} {tf} Window {start}-{end}] Strategy: {STRATEGY.__name__}")
                    print(f"  Train Sharpe: {train_metrics.get('train_sharpe', np.nan):.2f}, Test Sharpe: {test_metrics.get('test_sharpe', np.nan):.2f}")
                    print(f"  Train Winrate: {train_metrics.get('train_win_rate', np.nan):.1f}%, Test Winrate: {test_metrics.get('test_win_rate', np.nan):.1f}%")
                    print(f"  Train Balance: {train_metrics.get('train_final_balance', np.nan):.2f}, Test Balance: {test_metrics.get('test_final_balance', np.nan):.2f}")
                    print(f"  Train Profit: {train_metrics.get('train_abs_profit', np.nan):.2f} ({train_metrics.get('train_pct_profit', np.nan):.2f}%), "
                          f"Test Profit: {test_metrics.get('test_abs_profit', np.nan):.2f} ({test_metrics.get('test_pct_profit', np.nan):.2f}%)")
                    print(f"  Train MaxDD: {train_metrics.get('train_max_drawdown', np.nan):.2%}, Test MaxDD: {test_metrics.get('test_max_drawdown', np.nan):.2%}")
                    print(f"  Train Calmar: {train_metrics.get('train_calmar', np.nan):.2f}, Test Calmar: {test_metrics.get('test_calmar', np.nan):.2f}")
                    print(f"  Train Volatility: {train_metrics.get('train_volatility', np.nan):.2f}, Test Volatility: {test_metrics.get('test_volatility', np.nan):.2f}")
                    print(f"  Train Kelly: {train_metrics.get('train_kelly_criterion', np.nan):.2f}, Test Kelly: {test_metrics.get('test_kelly_criterion', np.nan):.2f}")
                    print(f"  Train Buy & Hold: {train_buyhold:.2f}% | Test Buy & Hold: {test_buyhold:.2f}%")
                    print(f"  Train Trades: {train_metrics.get('train_total_trades', 0)}, Test Trades: {test_metrics.get('test_total_trades', 0)}")
                    print("-" * 70)

                except Exception as e:
                    print(f"⚠️ Fejl i window {start}-{end} for {symbol} {tf}: {e}")
                    traceback.print_exc()
                    continue

            splits_count[(symbol, tf)] = splits
            print(f"  ➡️ Antal splits for {symbol} {tf}: {splits}")

            results_df = pd.DataFrame([r for r in all_results if r['symbol']==symbol and r['timeframe']==tf])
            if not results_df.empty:
                plot_path = plot_walkforward_results(results_df, symbol, tf)
                # Send graf til Telegram med Buy & Hold info
                bh_mean = results_df['test_buyhold_pct'].mean() if 'test_buyhold_pct' in results_df else np.nan
                send_image(
                    plot_path,
                    caption=f"Walkforward-graf for {symbol} {tf}\nBuy & Hold (test): {bh_mean:.2f}%"
                )

    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(
        OUTPUT_DIR, f"walkforward_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    if not results_df.empty:
        results_df.to_csv(out_csv, index=False)
        print(f"\n✅ Gemte walkforward-resultater til {out_csv}")
        send_document(
            out_csv,
            caption="Walkforward-summary (inkl. Buy & Hold)"
        )
        print("\n=== Top-10 vinduer efter out-of-sample (test) Sharpe ===")
        cols_to_show = [
            "symbol", "timeframe", "window_start", "window_end", "strategy",
            "test_sharpe", "test_win_rate", "test_final_balance", "test_pct_profit",
            "test_max_drawdown", "test_calmar", "test_volatility", "test_kelly_criterion",
            "test_buyhold_pct", "test_expectancy"
        ]
        show_cols = [col for col in cols_to_show if col in results_df.columns]
        print(results_df.sort_values("test_sharpe", ascending=False).head(10)[show_cols])
    else:
        print("Ingen walkforward-resultater blev genereret. Tjek feature-filer og pipeline.")

    print("\n--- Split count (vinduer pr. coin/timeframe): ---")
    for k, v in splits_count.items():
        print(f"{k}: {v} vinduer")
    print("\nSamlet antal splits:", sum(splits_count.values()))
