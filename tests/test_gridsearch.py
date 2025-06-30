# tests/test_gridsearch.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime
from config.config import COINS, TIMEFRAMES
from strategies.gridsearch_strategies import grid_search_sl_tp_ema, paper_trade_simple
from utils.performance import print_performance_report

OUTPUT_DIR = "outputs/gridsearch/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

# ---- Parametre til split ----
TEST_RATIO = 0.2  # 20% af data bruges til test/out-of-sample

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

        # -------- 1. Split i train/test (walk-forward) --------
        N = len(df)
        test_size = int(N * TEST_RATIO)
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

        # -------- 2. Grid search på train --------
        results_df = grid_search_sl_tp_ema(train_df)
        results_df["symbol"] = symbol
        results_df["timeframe"] = tf
        results_df["set"] = "train"

        # -------- 3. Vælg bedste strategi --------
        best = results_df.sort_values("sharpe", ascending=False).iloc[0]
        print("\nBedste parametre (in-sample):")
        print(best[["sharpe", "sl", "tp", "ema_fast", "ema_slow", "final_balance"]])

        # -------- 4. Evaluer på test (out-of-sample) --------
        # Regenerér EMA-kolonner og signal på test_df med bedste parametre
        efast, eslow = int(best["ema_fast"]), int(best["ema_slow"])
        test_df[f"ema_{efast}"] = test_df["close"].ewm(span=efast, adjust=False).mean()
        test_df[f"ema_{eslow}"] = test_df["close"].ewm(span=eslow, adjust=False).mean()
        rename_map = {
            f"ema_{efast}": "ema_9",
            f"ema_{eslow}": "ema_21"
        }
        strat_test_df = test_df.rename(columns=rename_map)
        strat_test_df = grid_search_sl_tp_ema.strategy_func(strat_test_df) if hasattr(grid_search_sl_tp_ema, "strategy_func") else paper_trade_simple(strat_test_df)[1]  # fallback

        # Backtest på test-split
        test_balance, test_trades = paper_trade_simple(strat_test_df, sl=best["sl"], tp=best["tp"])
        # Saml test-metrics
        from utils.performance import calculate_performance_metrics
        test_metrics = calculate_performance_metrics(test_trades["balance"], test_trades)
        test_metrics.update({
            "symbol": symbol, "timeframe": tf, "set": "test",
            "sl": best["sl"], "tp": best["tp"], "ema_fast": efast, "ema_slow": eslow,
            "final_balance": test_balance
        })

        # -------- 5. Gem alt --------
        out_csv = os.path.join(
            OUTPUT_DIR, f"gridsearch_{symbol.lower()}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_train.csv"
        )
        results_df.to_csv(out_csv, index=False)
        print(f"✅ Gemte gridsearch-resultater: {out_csv}")

        # Print top-5 for train
        top5 = results_df.sort_values("sharpe", ascending=False).head(5)
        print("\nTop-5 strategier (train):")
        print(top5[["sharpe", "sl", "tp", "ema_fast", "ema_slow", "final_balance"]])

        # Gem test resultater som DataFrame
        test_df_metrics = pd.DataFrame([test_metrics])
        all_results.append(results_df)
        all_results.append(test_df_metrics)

        # --- Rapporter out-of-sample på konsol ---
        print("\n=== Out-of-sample performance (test-split): ===")
        print_performance_report(test_trades["balance"], test_trades)

# Samlet oversigt – kan åbnes i Excel eller Pandas!
if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_ALL_train_test.csv"), index=False)
    print("\n=== Samlet top-10 Sharpe (train+test) på tværs af coins/timeframes ===")
    print(all_results_df.sort_values("sharpe", ascending=False).head(10)[
        ["symbol", "timeframe", "set", "sharpe", "sl", "tp", "ema_fast", "ema_slow", "final_balance"]
    ])
else:
    print("Ingen gridsearch-resultater blev genereret. Tjek feature-filer og pipeline.")

