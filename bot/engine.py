import sys
import os
import json
import pandas as pd
import numpy as np
import datetime
import glob

# === NYT: Importér ResourceMonitor til ressourceovervågning ===
from bot.monitor import ResourceMonitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from versions import (
        PIPELINE_VERSION, PIPELINE_COMMIT,
        FEATURE_VERSION, ENGINE_VERSION, ENGINE_COMMIT, MODEL_VERSION, LABEL_STRATEGY
    )
except ImportError:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = ENGINE_COMMIT = MODEL_VERSION = LABEL_STRATEGY = "unknown"

from models.model_training import train_model
from backtest.backtest import run_backtest, calc_backtest_metrics
from backtest.metrics import evaluate_strategies
from visualization.plot_backtest import plot_backtest
from visualization.plot_drawdown import plot_drawdown
from visualization.plot_strategy_score import plot_strategy_scores
from utils.telegram_utils import (
    send_image, send_message, send_performance_report
)
from utils.robust_utils import safe_run

# Ensemble/voting
from ensemble.majority_vote_ensemble import majority_vote_ensemble
from ensemble.weighted_vote_ensemble import weighted_vote_ensemble

# Klassiske strategier
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals
from strategies.ema_cross_strategy import ema_cross_signals

# Avancerede strategier
from strategies.advanced_strategies import (
    ema_crossover_strategy,
    ema_rsi_regime_strategy,
    ema_rsi_adx_strategy,
    rsi_mean_reversion,
    regime_ensemble,
    voting_ensemble,
    add_adaptive_sl_tp,
)

# Gridsearch
from strategies.gridsearch_strategies import grid_search_sl_tp_ema

from visualization.viz_feature_importance import plot_feature_importance
from utils.feature_logging import log_top_features_to_md, log_top_features_csv, send_top_features_telegram

try:
    from tuning.tuning_threshold import tune_threshold
except ImportError:
    tune_threshold = None

SYMBOL = "BTC"
GRAPH_DIR = "graphs/"
DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 0.7, 0.4, 1.0]  # Nu plads til 4 strategier
RETRAIN_WINRATE_THRESHOLD = 0.30
RETRAIN_PROFIT_THRESHOLD = 0.0
MAX_RETRAINS = 3

# === DEBUG/PROD: slå regime-filter til/fra ===
USE_REGIME_FILTER = False   # True = produktion, False = debug/test (ALTID test uden først!)
ADAPTIVE_WINRATE_THRESHOLD = 0.0  # Sæt lavt for at sikre altid aktivering for debug/test

def get_latest_csv(folder="outputs/feature_data/", pattern="*_features_*.csv"):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"Ingen datafiler fundet i {folder}")
    return max(files, key=os.path.getctime)

def load_best_ensemble_params(
    json_path="tuning/best_ensemble_params.json",
    txt_path="tuning/tuning_results_threshold.txt"
):
    threshold = DEFAULT_THRESHOLD
    weights = DEFAULT_WEIGHTS
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            threshold = data.get("threshold", DEFAULT_THRESHOLD)
            weights = data.get("weights", DEFAULT_WEIGHTS)
            print(f"[INFO] Indlæst tuning-parametre fra {json_path}: threshold={threshold}, weights={weights}")
            return threshold, weights
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke indlæse {json_path}: {e}")
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "Best threshold" in line:
                threshold = float(line.split(":")[1].strip())
            if "Best weights" in line:
                weights = eval(line.split(":")[1].strip())
        print(f"[INFO] Indlæst tuning-parametre fra {txt_path}: threshold={threshold}, weights={weights}")
    else:
        print(f"[INFO] Bruger default-parametre: threshold={threshold}, weights={weights}")
    return threshold, weights

def read_features_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    return df

def log_engine_meta(meta_path, feature_file, threshold, weights, strat_scores, metrics):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"run_time: {timestamp}\n")
        f.write(f"pipeline_version: {PIPELINE_VERSION}\n")
        f.write(f"pipeline_commit: {PIPELINE_COMMIT}\n")
        f.write(f"engine_version: {ENGINE_VERSION}\n")
        f.write(f"engine_commit: {ENGINE_COMMIT}\n")
        f.write(f"feature_version: {FEATURE_VERSION}\n")
        f.write(f"model_version: {MODEL_VERSION}\n")
        f.write(f"label_strategy: {LABEL_STRATEGY}\n")
        f.write(f"feature_file: {feature_file}\n")
        f.write(f"threshold: {threshold}\n")
        f.write(f"weights: {weights}\n")
        f.write(f"metrics: {json.dumps(metrics)}\n")
        f.write(f"strategy_scores: {json.dumps(strat_scores)}\n")
    print(f"📝 Engine meta logget til: {meta_path}")

def log_performance_metrics(metrics, filename="outputs/performance_metrics_history.csv"):
    import csv
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"✅ Metrics logget til {filename}")

def should_retrain(metrics):
    win_rate = metrics.get("win_rate", 0)
    profit_pct = metrics.get("profit_pct", 0)
    retrain = (win_rate < RETRAIN_WINRATE_THRESHOLD) or (profit_pct < RETRAIN_PROFIT_THRESHOLD)
    if retrain:
        print(f"🚨 Retrain trigget: win_rate={win_rate:.3f}, profit_pct={profit_pct:.3f}")
    return retrain

def main(threshold=DEFAULT_THRESHOLD, weights=DEFAULT_WEIGHTS, FORCE_DEBUG=False):
    # === START MONITOR HER ===
    monitor = ResourceMonitor(
        ram_max=85,              # Justér efter din maskine!
        cpu_max=90,
        gpu_max=95,
        gpu_temp_max=80,
        check_interval=10,
        action="pause",
        log_file="outputs/debug/resource_log.csv"
    )
    monitor.start()
    retrain_count = 0
    seed = None

    try:
        DATA_PATH = get_latest_csv()
        while True:
            print("🔄 Indlæser features:", DATA_PATH)
            df = read_features_auto(DATA_PATH)
            print(f"✅ Data indlæst ({len(df)} rækker)")
            print("Kolonner:", list(df.columns))

            # --- Check regime --- 
            if "regime" not in df.columns:
                msg = (
                    "❌ FEJL: Features-filen mangler kolonnen 'regime'.\n"
                    f"Kolonner fundet: {list(df.columns)}\n"
                    "Tip: Tjek feature engineering, og at alle steps køres i korrekt rækkefølge."
                )
                print(msg)
                send_message(msg)
                return

            regime_map = {"bull": "bull", "bear": "bear", "neutral": "neutral", 0: "bull", 1: "bear", 2: "neutral"}
            df["regime"] = df["regime"].map(regime_map).fillna(df["regime"])
            print("Regime-værdier i df:", df["regime"].value_counts(dropna=False).to_dict())

            print("🔄 Træner eller indlæser ML-model ...")
            model, model_path, feature_cols = train_model(df, random_seed=seed)
            print(f"✅ ML-model klar: {model_path}")
            X_pred = df[feature_cols]
            ml_raw = model.predict(X_pred)
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_pred)[:, 1]
                ml_signals = (probas > threshold).astype(int)
            else:
                ml_signals = ml_raw

            fi_path = None
            # === DEBUG/TESTMODE: FORCE TRADES ===
            if FORCE_DEBUG:
                print("‼️ DEBUG: Forcerer SKIFTEVIS BUY/SELL for test!")
                pattern = np.array([1, -1])
                ml_signals = np.tile(pattern, int(np.ceil(len(df)/2)))[:len(df)]
                rsi_signals = np.tile(pattern, int(np.ceil(len(df)/2)))[:len(df)]
                macd_signals = np.tile(pattern, int(np.ceil(len(df)/2)))[:len(df)]
                ema_signals = np.tile(pattern, int(np.ceil(len(df)/2)))[:len(df)]
            else:
                rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
                macd_signals = macd_cross_signals(df)
                ema_signals = ema_cross_signals(df)

            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    sorted_idx = np.argsort(imp)[::-1]
                    top_features = [(feature_cols[i], imp[i]) for i in sorted_idx[:15]]
                    fi_path = os.path.join(GRAPH_DIR, f"feature_importance_ML_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
                    plot_feature_importance(feature_cols, imp, out_path=fi_path, method="Permutation", top_n=15)
                    print(f"✅ Feature importance-plot gemt: {fi_path}")
                    log_top_features_to_md(top_features, md_path="BotStatus.md", model_name="ML")
                    log_top_features_csv(top_features, csv_path="data/top_features_history.csv", model_name="ML")
                    send_top_features_telegram(top_features, send_message, chat_id=None, model_name="ML")
            except Exception as e:
                print(f"⚠️ Fejl ved feature importance-plot eller log: {e}")

            print(f"Signal distribution ML/RSI/MACD/EMA:",
                  pd.Series(ml_signals).value_counts().to_dict(),
                  pd.Series(rsi_signals).value_counts().to_dict(),
                  pd.Series(macd_signals).value_counts().to_dict(),
                  pd.Series(ema_signals).value_counts().to_dict())

            print("Ensemble-weighted signal-fordeling:", pd.Series(weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, ema_signals, weights=weights)).value_counts().to_dict())

            # ENSEMBLE STRATEGI – du kan let tilføje flere strategier her!
            print(f"➡️  Bruger majority_vote_ensemble ...")
            majority_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals, ema_signals)
            print(f"➡️  Bruger weighted_vote_ensemble med weights: {weights}")
            weighted_signals = weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, ema_signals, weights=weights)
            df["signal"] = weighted_signals

            # --- GEM DEBUG-SIGNALER ---
            df_debug = df.copy()
            df_debug["ml_signal"] = ml_signals
            df_debug["rsi_signal"] = rsi_signals
            df_debug["macd_signal"] = macd_signals
            df_debug["ema_signal"] = ema_signals
            df_debug["ensemble_majority"] = majority_signals
            df_debug["ensemble_weighted"] = weighted_signals
            debug_cols = ["timestamp", "close", "ml_signal", "rsi_signal", "macd_signal", "ema_signal", "ensemble_majority", "ensemble_weighted"]
            df_debug[debug_cols].to_csv("outputs/signals_debug.csv", index=False)
            print("✅ Debug: signal-distribution gemt til outputs/signals_debug.csv")

            print("🔄 Kører backtest ...")
            trades_df, balance_df = run_backtest(df, signals=weighted_signals)
            metrics = calc_backtest_metrics(trades_df, balance_df)
            print("Backtest-metrics:", metrics)
            win_rate = metrics.get("win_rate", 0)
            profit_pct = metrics.get("profit_pct", 0)
            drawdown = metrics.get("drawdown_pct", 0)
            print(f"🔎 Win-rate: {win_rate*100:.2f}%, Profit: {profit_pct:.2f}%, Drawdown: {drawdown:.2f}%")

            log_performance_metrics(metrics)
            send_performance_report(metrics, symbol=SYMBOL, timeframe="1h", window=None)

            # --- AVANCERET STRATEGI-EVALUERING ---
            strat_scores = evaluate_strategies(
                df=df,
                ml_signals=ml_signals,
                rsi_signals=rsi_signals,
                macd_signals=macd_signals,
                ensemble_signals=weighted_signals,
                trades_df=trades_df,
                balance_df=balance_df
            )
            print("Strategi-score:", strat_scores)

            regime_stats = strat_scores["ENSEMBLE"].get("regime_stats", {})
            print("DEBUG regime_stats:", json.dumps(regime_stats, indent=2))
            active_regimes = []

            if USE_REGIME_FILTER and regime_stats:
                for regime, stats in regime_stats.items():
                    print(f"  Regime: {regime}, win_rate: {stats.get('win_rate')}, trades: {stats.get('num_trades')}")
                    if stats.get("win_rate", 0) >= ADAPTIVE_WINRATE_THRESHOLD:
                        active_regimes.append(regime)
                print(f"Aktive regimer for handel: {active_regimes}")

                filtered_signals = []
                for idx, row in df.iterrows():
                    this_regime = str(row.get("regime", ""))
                    if this_regime in active_regimes:
                        filtered_signals.append(df.at[idx, "signal"])
                    else:
                        filtered_signals.append(0)
                df["signal"] = filtered_signals

                trades_df, balance_df = run_backtest(df, signals=df["signal"].values)
                metrics = calc_backtest_metrics(trades_df, balance_df)
                print(f"[ADAPTIV] Backtest-metrics efter regime-filter:", metrics)
                win_rate = metrics.get("win_rate", 0)
                profit_pct = metrics.get("profit_pct", 0)
                drawdown = metrics.get("drawdown_pct", 0)
                print(f"[ADAPTIV] Win-rate: {win_rate*100:.2f}%, Profit: {profit_pct:.2f}%, Drawdown: {drawdown:.2f}%")
                log_performance_metrics(metrics)
                send_message(
                    f"🤖 Adaptiv regime-strategi aktiv!\n"
                    f"Aktive regimer: {active_regimes}\n"
                    f"Profit: {profit_pct:.2f}% | Win-rate: {win_rate*100:.1f}% | Trades: {metrics.get('num_trades', 'N/A')}\n"
                )
                send_performance_report(metrics, symbol=SYMBOL, timeframe="1h", window="ADAPTIV")
            else:
                print("🚧 Regime-filter slået FRA – alle signaler bruges!")

            meta_path = f"outputs/feature_data/engine_meta_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
            log_engine_meta(
                meta_path=meta_path,
                feature_file=DATA_PATH,
                threshold=threshold,
                weights=weights,
                strat_scores=strat_scores,
                metrics=metrics
            )

            score_plot_path = os.path.join(
                GRAPH_DIR, f"strategy_scores_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
            )
            plot_strategy_scores(strat_scores, save_path=score_plot_path)
            print(f"✅ Strategi-score-graf gemt: {score_plot_path}")

            print("🔄 Genererer grafer ...")
            plot_path = plot_backtest(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)
            drawdown_path = plot_drawdown(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)

            print("🔄 Sender grafer til Telegram ...")
            send_message(
                f"✅ Backtest for {SYMBOL} afsluttet!\n"
                f"Mode: Ensemble voting\n"
                f"Weights: {weights}\n"
                f"Threshold: {threshold}\n"
                f"Profit: {profit_pct:.2f}% | Win-rate: {win_rate*100:.1f}% | Trades: {metrics.get('num_trades', 'N/A')}\n"
                f"\n"
                f"📊 Strategi-score:\n"
                f"ML:    {strat_scores['ML']}\n"
                f"RSI:   {strat_scores['RSI']}\n"
                f"MACD:  {strat_scores['MACD']}\n"
                f"Ensemble: {strat_scores['ENSEMBLE']}\n"
                f"\n"
                f"Versionsinfo: pipeline {PIPELINE_VERSION}/{PIPELINE_COMMIT}, engine {ENGINE_VERSION}/{ENGINE_COMMIT}, model {MODEL_VERSION}, feature {FEATURE_VERSION}"
            )
            send_image(plot_path, caption=f"📈 Balanceudvikling for {SYMBOL}")
            send_image(drawdown_path, caption=f"📉 Drawdown for {SYMBOL}")
            send_image(score_plot_path, caption="📊 Strategi-score ML/RSI/MACD/EMA/Ensemble")
            if fi_path is not None:
                send_image(fi_path, caption="🧠 Feature Importance for ML-model")

            if should_retrain(metrics):
                retrain_count += 1
                if retrain_count > MAX_RETRAINS:
                    send_message(f"⚠️ Maksimalt antal retrains nået ({MAX_RETRAINS}). Stopper retrain-loop.")
                    print("⚠️ Maksimalt antal retrains nået. Afslutter loop.")
                    break
                send_message(
                    f"🚨 Retrain trigget automatisk! Win-rate: {win_rate*100:.1f}%, Profit: {profit_pct:.2f}% – Starter retrain (forsøg {retrain_count})"
                )
                print(f"🚨 Retrain trigget! Starter ny træning (forsøg {retrain_count}) ...")
                seed = np.random.randint(0, 100_000)
                continue
            else:
                print("✅ Performance er tilfredsstillende – ingen retrain nødvendig.")
                break

        print("🎉 Hele flowet er nu automatisk!")
    finally:
        monitor.stop()

if __name__ == "__main__":
    threshold, weights = load_best_ensemble_params()
    safe_run(lambda: main(threshold=threshold, weights=weights, FORCE_DEBUG=False))
