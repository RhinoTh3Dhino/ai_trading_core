from utils.project_path import PROJECT_ROOT
# tests/test_walkforward.py

# 📌 Sikrer korrekt sys.path til projektroden
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ Korrekte imports
import glob
import traceback
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# === Centralt styrede konfigurationer ===
try:
    from config.monitoring_config import (
        COINS,
        TIMEFRAMES,
        WALKFORWARD_DEFAULT_WINDOW_SIZE,
        WALKFORWARD_MIN_WINDOW_SIZE,
        WALKFORWARD_STEP_SIZE,
        WALKFORWARD_TRAIN_SIZE,
        ENABLE_MONITORING,
        ALARM_THRESHOLDS,
    )
except ImportError:
    COINS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
    TIMEFRAMES = ["1h", "4h"]
    WALKFORWARD_DEFAULT_WINDOW_SIZE = 200
    WALKFORWARD_MIN_WINDOW_SIZE = 100
    WALKFORWARD_STEP_SIZE = 50
    WALKFORWARD_TRAIN_SIZE = 0.7
    ENABLE_MONITORING = True
    ALARM_THRESHOLDS = {"drawdown": -20, "winrate": 20, "profit": -10}

from strategies.advanced_strategies import (
    ema_crossover_strategy,
    ema_rsi_regime_strategy,
    voting_ensemble,
)
from bot.paper_trader import paper_trade as paper_trade_advanced
from strategies.gridsearch_strategies import paper_trade_simple
from utils.performance import (
    calculate_performance_metrics,
    calculate_rolling_sharpe,
    calculate_trade_duration,
    calculate_regime_drawdown,
)
from utils.telegram_utils import send_image, send_document


# --- WALKFORWARD PARAMS (hentet fra config hvis muligt) ---
DEFAULT_WINDOW_SIZE = WALKFORWARD_DEFAULT_WINDOW_SIZE
MIN_WINDOW_SIZE = WALKFORWARD_MIN_WINDOW_SIZE
STEP_SIZE = WALKFORWARD_STEP_SIZE
TRAIN_SIZE = WALKFORWARD_TRAIN_SIZE

# AUTO PATH CONVERTED
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "walkforward/"
BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# AUTO PATH CONVERTED
def get_latest_feature_file(symbol, timeframe, feature_dir=PROJECT_ROOT / "outputs" / "feature_data"):
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

def flatten_dict(d, prefix=""):
    flat = {}
    if isinstance(d, dict):
        for k, v in d.items():
            flat[f"{prefix}{k}"] = v if np.isscalar(v) or v is None else 0.0
    return flat

def plot_walkforward_results(results_df, symbol, tf):
    windows = range(len(results_df))
    plt.figure(figsize=(14,6))
    plt.plot(windows, results_df['test_sharpe'], label='Test Sharpe')
    plt.plot(windows, results_df['test_win_rate'], label='Test Win-rate (%)')
    plt.plot(windows, results_df['test_final_balance'], label='Test Balance')
    plt.plot(windows, results_df['test_pct_profit'], label='Test Profit %')
    plt.plot(windows, results_df['test_max_drawdown'], label='Test Max Drawdown')
    if 'test_volatility' in results_df:
        plt.plot(windows, results_df['test_volatility'], label='Test Volatility')
    if 'test_calmar' in results_df:
        plt.plot(windows, results_df['test_calmar'], label='Test Calmar')
    if 'test_buyhold_pct' in results_df:
        plt.plot(windows, results_df['test_buyhold_pct'], label='Buy & Hold (test)')
    if 'test_rolling_sharpe' in results_df:
        plt.plot(windows, results_df['test_rolling_sharpe'], label='Rolling Sharpe')
    plt.title(f"Walkforward Performance for {symbol} {tf}")
    plt.xlabel("Walkforward Window")
    plt.legend()
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}walkforward_plot_{symbol}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
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

                    # Rolling Sharpe
                    try:
                        train_metrics['rolling_sharpe'] = calculate_rolling_sharpe(train_trades["balance"], window=50)
                    except Exception:
                        train_metrics['rolling_sharpe'] = np.nan
                    try:
                        test_metrics['rolling_sharpe'] = calculate_rolling_sharpe(test_trades["balance"], window=50)
                    except Exception:
                        test_metrics['rolling_sharpe'] = np.nan

                    # Trade duration
                    try:
                        td_train = calculate_trade_duration(train_trades)
                        for key, value in td_train.items():
                            train_metrics[key] = value
                    except Exception:
                        train_metrics['mean_trade_duration'] = 0.0
                        train_metrics['median_trade_duration'] = 0.0
                        train_metrics['max_trade_duration'] = 0.0
                    try:
                        td_test = calculate_trade_duration(test_trades)
                        for key, value in td_test.items():
                            test_metrics[key] = value
                    except Exception:
                        test_metrics['mean_trade_duration'] = 0.0
                        test_metrics['median_trade_duration'] = 0.0
                        test_metrics['max_trade_duration'] = 0.0

                    # Regime drawdown
                    try:
                        reg_dd_train = calculate_regime_drawdown(train_trades)
                        train_metrics.update(flatten_dict(reg_dd_train, "regime_"))
                    except Exception:
                        pass
                    try:
                        reg_dd_test = calculate_regime_drawdown(test_trades)
                        test_metrics.update(flatten_dict(reg_dd_test, "regime_"))
                    except Exception:
                        pass

                    train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

                    # Buy & Hold
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
                    print(f"  Train Profit: {train_metrics.get('train_pct_profit', np.nan):.2f}%, Test Profit: {test_metrics.get('test_pct_profit', np.nan):.2f}%")
                    print(f"  Train Rolling Sharpe: {train_metrics.get('train_rolling_sharpe', np.nan)}, Test Rolling Sharpe: {test_metrics.get('test_rolling_sharpe', np.nan)}")
                    print(f"  Train Mean Trade Duration: {train_metrics.get('train_mean_trade_duration', np.nan)}, Test Mean Trade Duration: {test_metrics.get('test_mean_trade_duration', np.nan)}")
                    print(f"  Train Regime Drawdown: {train_metrics.get('train_regime_drawdown_bull', np.nan)}, Test Regime Drawdown: {test_metrics.get('test_regime_drawdown_bull', np.nan)}")
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
                bh_mean = results_df['test_buyhold_pct'].mean() if 'test_buyhold_pct' in results_df else np.nan
                send_image(
                    plot_path,
                    caption=f"Walkforward-graf for {symbol} {tf}\nBuy & Hold (test): {bh_mean:.2f}%"
                )

    # --- Robusthed: Fjern inf og NaN, erstat med 0 ---
    results_df = pd.DataFrame(all_results)
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    results_df = results_df.fillna(0)

    # --- Markér bedste split og top-5 splits ---
    results_df["is_best_split"] = False
    results_df["is_top5_split"] = False
    if not results_df.empty and "test_sharpe" in results_df.columns:
        # Best split (højeste test_sharpe)
        best_idx = results_df["test_sharpe"].idxmax()
        results_df.loc[best_idx, "is_best_split"] = True
        # Top 5 splits (efter test_sharpe)
        top5_idx = results_df.sort_values("test_sharpe", ascending=False).head(5).index
        results_df.loc[top5_idx, "is_top5_split"] = True

    out_csv = os.path.join(
        OUTPUT_DIR, f"walkforward_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    if not results_df.empty:
        # Gem CSV
        results_df.to_csv(out_csv, index=False)
        print(f"\n✅ Gemte walkforward-resultater til {out_csv}")

        # Ekstra eksport
        excel_path = out_csv.replace(".csv", ".xlsx")
        json_path = out_csv.replace(".csv", ".json")
        results_df.to_excel(excel_path, index=False)
        results_df.to_json(json_path, orient="records")
        print(f"✅ Gemte også som Excel: {excel_path}")
        print(f"✅ Gemte også som JSON: {json_path}")

        # Backup alle filer
        for f in [out_csv, excel_path, json_path]:
            backup_f = os.path.join(BACKUP_DIR, os.path.basename(f))
            shutil.copy(f, backup_f)
            print(f"🔄 Backup gemt: {backup_f}")

        # === Gem og backup top-5/top-10 splits ===
        for N, tag in [(5, "top5"), (10, "top10")]:
            topN = results_df.sort_values("test_sharpe", ascending=False).head(N)
            topN_csv = out_csv.replace(".csv", f"_{tag}_splits.csv")
            topN_excel = out_csv.replace(".csv", f"_{tag}_splits.xlsx")
            topN_json = out_csv.replace(".csv", f"_{tag}_splits.json")
            # Gem
            topN.to_csv(topN_csv, index=False)
            topN.to_excel(topN_excel, index=False)
            topN.to_json(topN_json, orient="records")
            print(f"✅ Gemte {tag} splits til: {topN_csv}, {topN_excel}, {topN_json}")
            # Backup
            for f in [topN_csv, topN_excel, topN_json]:
                backup_f = os.path.join(BACKUP_DIR, os.path.basename(f))
                shutil.copy(f, backup_f)
                print(f"🔄 Backup gemt: {backup_f}")
            # Telegram
            send_document(topN_csv, caption=f"{tag.upper()} splits (CSV)")
            send_document(topN_excel, caption=f"{tag.upper()} splits (Excel)")
            send_document(topN_json, caption=f"{tag.upper()} splits (JSON)")

        # Telegram-eksport af summary
        send_document(
            out_csv,
            caption="Walkforward-summary (inkl. Buy & Hold, avancerede metrics, best/top5 markeringer)"
        )
        send_document(
            excel_path,
            caption="Walkforward-summary (Excel)"
        )
        send_document(
            json_path,
            caption="Walkforward-summary (JSON)"
        )

        print("\n=== Top-10 vinduer efter out-of-sample (test) Sharpe ===")
        cols_to_show = [
            "symbol", "timeframe", "window_start", "window_end", "strategy",
            "test_sharpe", "test_win_rate", "test_final_balance", "test_pct_profit",
            "test_max_drawdown", "test_calmar", "test_volatility", "test_kelly_criterion",
            "test_buyhold_pct", "test_rolling_sharpe",
            "test_mean_trade_duration", "test_median_trade_duration", "test_max_trade_duration",
            "is_best_split", "is_top5_split"
        ]
        show_cols = [col for col in cols_to_show if col in results_df.columns]
        print(results_df.sort_values("test_sharpe", ascending=False).head(10)[show_cols])
    else:
        print("Ingen walkforward-resultater blev genereret. Tjek feature-filer og pipeline.")

    print("\n--- Split count (vinduer pr. coin/timeframe): ---")
    for k, v in splits_count.items():
        print(f"{k}: {v} vinduer")
    print("\nSamlet antal splits:", sum(splits_count.values()))