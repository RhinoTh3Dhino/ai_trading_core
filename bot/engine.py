import sys, os
import json
import pandas as pd
import numpy as np
import datetime
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Versionsinfo fra versions.py ---
try:
    from versions import (
        PIPELINE_VERSION, PIPELINE_COMMIT,
        FEATURE_VERSION, ENGINE_VERSION, ENGINE_COMMIT, MODEL_VERSION, LABEL_STRATEGY
    )
except ImportError:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = ENGINE_COMMIT = MODEL_VERSION = LABEL_STRATEGY = "unknown"

# MODEL & STRATEGI-IMPORTS
from models.model_training import train_model
from backtest.backtest import run_backtest, calc_backtest_metrics
from backtest.metrics import evaluate_strategies
from visualization.plot_backtest import plot_backtest
from visualization.plot_drawdown import plot_drawdown
from visualization.plot_strategy_score import plot_strategy_scores
from utils.telegram_utils import (
    send_image, send_message,
    send_regime_summary, send_regime_warning
)
from utils.robust_utils import safe_run
from ensemble.majority_vote_ensemble import majority_vote_ensemble
from ensemble.weighted_vote_ensemble import weighted_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

# FEATURE IMPORTANCE LOGNING
from visualization.viz_feature_importance import plot_feature_importance
from utils.feature_logging import (
    log_top_features_to_md,
    log_top_features_csv,
    send_top_features_telegram,
)

# Optuna-tuning (valgfri)
try:
    from tuning.tuning_threshold import tune_threshold
except ImportError:
    tune_threshold = None

SYMBOL = "BTC"
GRAPH_DIR = "graphs/"
DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHTS = [1.0, 0.7, 0.4]
ADAPTIVE_WINRATE_THRESHOLD = 0.3  # Win-rate-grænse for aktiv regime

def get_latest_csv(folder="outputs/feature_data/", pattern="btc_1h_features_*.csv"):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError("Ingen datafiler fundet i " + folder)
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
                weight_str = line.split(":")[1].strip()
                weights = eval(weight_str)
        print(f"[INFO] Indlæst tuning-parametre fra {txt_path}: threshold={threshold}, weights={weights}")
    else:
        print(f"[INFO] Bruger default-parametre: threshold={threshold}, weights={weights}")
    return threshold, weights

def read_features_auto(file_path):
    """Indlæs features-CSV – spring meta-header over hvis nødvendigt."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    return df

def log_engine_meta(meta_path, feature_file, threshold, weights, strat_scores, metrics):
    """Gem versionsinfo og runparametre for engine/run"""
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

def main(threshold=DEFAULT_THRESHOLD, weights=DEFAULT_WEIGHTS):
    DATA_PATH = get_latest_csv()
    print("🔄 Indlæser features:", DATA_PATH)
    df = read_features_auto(DATA_PATH)
    print(f"✅ Data indlæst ({len(df)} rækker)")
    print("Kolonner:", list(df.columns))

    # Robust check: Er 'regime' til stede?
    if "regime" not in df.columns:
        msg = (
            "❌ FEJL: Features-filen mangler kolonnen 'regime'.\n"
            f"Kolonner fundet: {list(df.columns)}\n"
            "Tip: Tjek feature engineering, og at alle steps køres i korrekt rækkefølge."
        )
        print(msg)
        send_message(msg)
        return

    # Robust regime-mapping
    regime_map = {0: "bull", 1: "bear", 2: "neutral"}
    df["regime"] = df["regime"].map(regime_map).fillna(df["regime"])
    print("Regime-værdier i df:", df["regime"].value_counts(dropna=False).to_dict())

    # ML-model træning & prediction
    print("🔄 Træner eller indlæser ML-model ...")
    model, model_path, feature_cols = train_model(df)
    print(f"✅ ML-model klar: {model_path}")
    X_pred = df[feature_cols]
    ml_raw = model.predict(X_pred)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_pred)[:, 1]
        ml_signals = (probas > threshold).astype(int)
    else:
        ml_signals = ml_raw

    # FEATURE IMPORTANCE, LOG & TELEGRAM
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            sorted_idx = np.argsort(imp)[::-1]
            top_features = [(feature_cols[i], imp[i]) for i in sorted_idx[:5]]
            fi_path = os.path.join(GRAPH_DIR, f"feature_importance_ML_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
            plot_feature_importance(feature_cols, imp, out_path=fi_path, method="Permutation", top_n=15)
            print(f"✅ Feature importance-plot gemt: {fi_path}")
            log_top_features_to_md(top_features, md_path="BotStatus.md", model_name="ML")
            log_top_features_csv(top_features, csv_path="data/top_features_history.csv", model_name="ML")
            send_top_features_telegram(top_features, send_message, chat_id=None, model_name="ML")
    except Exception as e:
        print(f"⚠️ Fejl ved feature importance-plot eller log: {e}")

    # Indikator-strategier
    print("🔄 Genererer strategi-signaler ...")
    rsi_signals = rsi_rule_based_signals(df, low=30, high=70)
    macd_signals = macd_cross_signals(df)
    print(f"Signal distribution ML/RSI/MACD:",
          pd.Series(ml_signals).value_counts().to_dict(),
          pd.Series(rsi_signals).value_counts().to_dict(),
          pd.Series(macd_signals).value_counts().to_dict())

    # Ensemble voting (vægtet)
    print(f"➡️  Bruger vægtet voting med weights: {weights}")
    ensemble_signals = weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, weights=weights)
    df["signal"] = ensemble_signals

    # Backtest
    print("🔄 Kører backtest ...")
    trades_df, balance_df = run_backtest(df, signals=ensemble_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    print("Backtest-metrics:", metrics)
    # Udtræk og print de vigtigste performance-metrics
    win_rate = metrics.get("win_rate", 0)
    profit_pct = metrics.get("profit_pct", 0)
    drawdown = metrics.get("max_drawdown", 0)
    print(f"🔎 Win-rate: {win_rate*100:.2f}%, Profit: {profit_pct}%, Drawdown: {drawdown}%")

    # Log alle metrics til performance-history (step 3A)
    log_performance_metrics(metrics)

    # Strategi-score på tværs af signaler (inkl. ROBUST regime-analyse via metrics.py)
    strat_scores = evaluate_strategies(
        df=df,
        ml_signals=ml_signals,
        rsi_signals=rsi_signals,
        macd_signals=macd_signals,
        ensemble_signals=ensemble_signals,
        trades_df=trades_df,
        balance_df=balance_df
    )
    print("Strategi-score:", strat_scores)

    # -------- ADAPTIV REGIME-FILTER --------
    regime_stats = strat_scores["ENSEMBLE"].get("regime_stats", {})
    active_regimes = []
    if regime_stats:
        for regime, stats in regime_stats.items():
            if stats["win_rate"] >= ADAPTIVE_WINRATE_THRESHOLD:
                active_regimes.append(regime)
        print(f"Aktive regimer for handel: {active_regimes}")

        # Kun handle i aktive regimer
        filtered_signals = []
        for idx, row in df.iterrows():
            this_regime = str(row.get("regime", ""))
            if this_regime in active_regimes:
                filtered_signals.append(df.at[idx, "signal"])
            else:
                filtered_signals.append(0)  # HOLD
        df["signal"] = filtered_signals

        trades_df, balance_df = run_backtest(df, signals=df["signal"].values)
        metrics = calc_backtest_metrics(trades_df, balance_df)
        print(f"[ADAPTIV] Backtest-metrics efter regime-filter:", metrics)
        win_rate = metrics.get("win_rate", 0)
        profit_pct = metrics.get("profit_pct", 0)
        drawdown = metrics.get("max_drawdown", 0)
        print(f"[ADAPTIV] Win-rate: {win_rate*100:.2f}%, Profit: {profit_pct}%, Drawdown: {drawdown}%")
        log_performance_metrics(metrics)
        send_message(
            f"🤖 Adaptiv regime-strategi aktiv!\n"
            f"Aktive regimer: {active_regimes}\n"
            f"Profit: {profit_pct}% | Win-rate: {win_rate*100:.1f}% | Trades: {metrics['num_trades']}\n"
        )
    else:
        print("Ingen regime-stats fundet – adaptiv strategi springes over.")

    # --- LOG versionsinfo og runparametre ---
    meta_path = f"outputs/feature_data/engine_meta_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    log_engine_meta(
        meta_path=meta_path,
        feature_file=DATA_PATH,
        threshold=threshold,
        weights=weights,
        strat_scores=strat_scores,
        metrics=metrics
    )

    # Visualisering af strategi-score
    score_plot_path = os.path.join(
        GRAPH_DIR, f"strategy_scores_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    )
    plot_strategy_scores(strat_scores, save_path=score_plot_path)
    print(f"✅ Strategi-score-graf gemt: {score_plot_path}")

    # Balance-graf og drawdown-graf
    print("🔄 Genererer grafer ...")
    plot_path = plot_backtest(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)
    drawdown_path = plot_drawdown(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)

    # Telegram (inkluder strategi-score og graf)
    print("🔄 Sender grafer til Telegram ...")
    send_message(
        f"✅ Backtest for {SYMBOL} afsluttet!\n"
        f"Mode: Weighted voting\n"
        f"Weights: {weights}\n"
        f"Threshold: {threshold}\n"
        f"Profit: {profit_pct}% | Win-rate: {win_rate*100:.1f}% | Trades: {metrics['num_trades']}\n"
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
    send_image(score_plot_path, caption="📊 Strategi-score ML/RSI/MACD/Ensemble")
    if 'fi_path' in locals():
        send_image(fi_path, caption="🧠 Feature Importance for ML-model")

    print("🎉 Hele flowet er nu automatisk!")

if __name__ == "__main__":
    if "--tune" in sys.argv and tune_threshold:
        send_message("🔧 Starter automatisk tuning af threshold og weights...")
        best_threshold, best_weights = tune_threshold()
        send_message(
            f"🏆 Bedste fundne threshold: {best_threshold:.3f}, weights: {best_weights} – genstarter backtest med nye værdier."
        )
        safe_run(lambda: main(threshold=best_threshold, weights=best_weights))
    else:
        threshold, weights = load_best_ensemble_params()
        safe_run(lambda: main(threshold=threshold, weights=weights))
