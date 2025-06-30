# tests/test_walkforward.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime
from config.config import COINS, TIMEFRAMES
from strategies.advanced_strategies import ema_crossover_strategy
from strategies.gridsearch_strategies import paper_trade_simple
from utils.performance import calculate_performance_metrics

# --- Walkforward-parametre ---
DEFAULT_WINDOW_SIZE = 1000    # Standard vindue, men tilpasses dynamisk
MIN_WINDOW_SIZE = 200         # Minimum rows for et vindue
STEP_SIZE = 250               # Hvor meget vinduet rulles frem for hver walk
TRAIN_SIZE = 0.7              # Procent af vinduet til træning

OUTPUT_DIR = "outputs/walkforward/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def walkforward_split(df, window_size, train_size, step_size):
    n = len(df)
    if n < window_size:
        # Gør vinduet så stort som muligt, men mindst MIN_WINDOW_SIZE
        if n >= MIN_WINDOW_SIZE:
            window_size = n
        else:
            return  # Yield intet
    train_len = int(window_size * train_size)
    test_len = window_size - train_len

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        train_df = df.iloc[start : start + train_len]
        test_df = df.iloc[start + train_len : end]
        yield train_df, test_df, start, end

if __name__ == "__main__":
    all_results = []
    splits_count = {}

    for symbol in COINS:
        for tf in TIMEFRAMES:
            feature_path = f"outputs/feature_data/{symbol.lower()}_{tf}_features_v1.3_{datetime.now().strftime('%Y%m%d')}.csv"
            if not os.path.exists(feature_path):
                print(f"❌ Feature-fil mangler: {feature_path}")
                continue

            print(f"\n=== Walkforward Validation: {symbol} {tf} ===")
            df = pd.read_csv(feature_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            n = len(df)
            splits = 0

            # Brug automatisk window size hvis datasæt er lille
            window_size = min(DEFAULT_WINDOW_SIZE, n) if n >= MIN_WINDOW_SIZE else 0
            if window_size < MIN_WINDOW_SIZE:
                print(f"⚠️  Ikke nok data ({n} rækker) til walkforward på {symbol} {tf}. Skipper...")
                continue

            for train_df, test_df, start, end in walkforward_split(df, window_size, TRAIN_SIZE, STEP_SIZE):
                splits += 1
                # Træn og test på hver split
                train_df = ema_crossover_strategy(train_df)
                test_df = ema_crossover_strategy(test_df)
                train_balance, train_trades = paper_trade_simple(train_df)
                train_metrics = calculate_performance_metrics(train_trades["balance"], train_trades)
                train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                test_balance, test_trades = paper_trade_simple(test_df)
                test_metrics = calculate_performance_metrics(test_trades["balance"], test_trades)
                test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
                result = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "window_start": start,
                    "window_end": end,
                }
                result.update(train_metrics)
                result.update(test_metrics)
                all_results.append(result)
                # Udskriv summary for hver periode
                print(f"[{symbol} {tf} Window {start}-{end}] Train Sharpe: {train_metrics['train_sharpe']:.2f}, Test Sharpe: {test_metrics['test_sharpe']:.2f}")
                print(f"  Train Winrate: {train_metrics['train_win_rate']:.1f}%, Test Winrate: {test_metrics['test_win_rate']:.1f}%")
                print(f"  Train Balance: {train_metrics['train_final_balance']:.2f}, Test Balance: {test_metrics['test_final_balance']:.2f}")
            splits_count[(symbol, tf)] = splits

    # Samlet DataFrame med alle vinduer og metrics
    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(
        OUTPUT_DIR, f"walkforward_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    if not results_df.empty:
        results_df.to_csv(out_csv, index=False)
        print(f"\n✅ Gemte walkforward-resultater til {out_csv}")
        print("\n=== Top-10 vinduer efter out-of-sample (test) Sharpe ===")
        print(results_df.sort_values("test_sharpe", ascending=False).head(10)[
            ["symbol", "timeframe", "window_start", "window_end", "test_sharpe", "test_win_rate", "test_final_balance"]
        ])
    else:
        print("Ingen walkforward-resultater blev genereret. Tjek feature-filer og pipeline.")
    print("\n--- Split count (vinduer pr. coin/timeframe): ---")
    for k, v in splits_count.items():
        print(f"{k}: {v} vinduer")
