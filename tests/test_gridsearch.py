# test_gridsearch.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime
from config.config import COINS, TIMEFRAMES
from strategies.gridsearch_strategies import grid_search_sl_tp_ema

OUTPUT_DIR = "outputs/gridsearch/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

for symbol in COINS:
    for tf in TIMEFRAMES:
        feature_path = f"outputs/feature_data/{symbol.lower()}_{tf}_features_v1.3_{datetime.now().strftime('%Y%m%d')}.csv"
        if not os.path.exists(feature_path):
            print(f"❌ Feature-fil mangler: {feature_path}")
            continue

        print(f"\n=== Grid Search: {symbol} {tf} ===")
        df = pd.read_csv(feature_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Kør grid search (ændr evt. ranges i grid_search_sl_tp_ema for at eksperimentere!)
        results_df = grid_search_sl_tp_ema(df)
        results_df["symbol"] = symbol
        results_df["timeframe"] = tf

        # Gem CSV for hvert run
        out_csv = os.path.join(
            OUTPUT_DIR, f"gridsearch_{symbol.lower()}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(out_csv, index=False)
        print(f"✅ Gemte gridsearch-resultater: {out_csv}")

        # Print top-5
        top5 = results_df.sort_values("sharpe", ascending=False).head(5)
        print("\nTop-5 strategier:")
        print(top5[["sharpe", "sl", "tp", "ema_fast", "ema_slow", "final_balance"]])
        all_results.append(results_df)

# Samlet oversigt – kan åbnes i Excel eller Pandas!
if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_ALL.csv"), index=False)
    print("\n=== Samlet top-10 på tværs af coins/timeframes ===")
    print(all_results_df.sort_values("sharpe", ascending=False).head(10)[
        ["symbol", "timeframe", "sharpe", "sl", "tp", "ema_fast", "ema_slow", "final_balance"]
    ])
else:
    print("Ingen gridsearch-resultater blev genereret. Tjek feature-filer og pipeline.")
