import json
import logging
import os
from datetime import datetime

import optuna
import pandas as pd
from dotenv import load_dotenv

from backtest.backtest import calc_backtest_metrics, run_backtest
from ensemble.majority_vote_ensemble import weighted_vote_ensemble
from models.model_training import train_model
from strategies.macd_strategy import macd_cross_signals
from strategies.rsi_strategy import rsi_rule_based_signals
from utils.project_path import PROJECT_ROOT

# === Load miljøvariabler fra .env ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def telegram_enabled():
    """Returnerer True hvis Telegram er korrekt konfigureret (og ikke dummy i CI)."""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.lower() in ("", "none", "dummy_token"):
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID.lower() in ("", "none", "dummy_id"):
        return False
    return True


def send_telegram_message(message):
    if telegram_enabled():
        try:
            from telegram import Bot

            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            logger = get_tuning_logger()
            logger.error(f"Telegram-fejl: {e}")
    else:
        print(f"🔕 [CI/test] Ville have sendt Telegram-besked: {message}")


# === Konstanter til tuning ===
# AUTO PATH CONVERTED
DATA_PATH = (
    PROJECT_ROOT / "outputs" / "feature_data/btc_1h_features_v_test_20250610.csv"
)
SYMBOL = "BTC"
TUNER_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(TUNER_DIR, "tuning_log.txt")
RESULTS_PATH = os.path.join(TUNER_DIR, "tuning_results_threshold.txt")
SNAPSHOT_PATH = os.path.join(TUNER_DIR, "best_ensemble_params.json")

# --- Sikr at tuning-mappen findes ---
if not os.path.exists(TUNER_DIR):
    os.makedirs(TUNER_DIR)

print(f"[INFO] Logfil skrives til: {LOG_PATH}")
print(f"[INFO] Resultatfil skrives til: {RESULTS_PATH}")


# === Logger-setup ===
def get_tuning_logger():
    logger = logging.getLogger("tuning_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_PATH, mode="w")
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# === OPTUNA-OBJECTIVE: Tune threshold OG weights ===
def objective(trial):
    logger = get_tuning_logger()
    threshold = trial.suggest_float("threshold", 0.5, 0.9)
    weight_ml = trial.suggest_float("weight_ml", 0.0, 2.0)
    weight_rsi = trial.suggest_float("weight_rsi", 0.0, 2.0)
    weight_macd = trial.suggest_float("weight_macd", 0.0, 2.0)
    weights = [weight_ml, weight_rsi, weight_macd]

    df = pd.read_csv(DATA_PATH)
    model, _, feature_cols = train_model(df)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df[feature_cols])[:, 1]
        ml_signals = (probas > threshold).astype(int)
    else:
        ml_signals = model.predict(df[feature_cols])

    rsi_signals = rsi_rule_based_signals(df, low=30, high=70)
    macd_signals = macd_cross_signals(df)
    ensemble_signals = weighted_vote_ensemble(
        ml_signals, rsi_signals, macd_signals, weights=weights
    )
    df["signal"] = ensemble_signals
    trades_df, balance_df = run_backtest(df, signals=ensemble_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    log_line = (
        f"{datetime.now()} | threshold={threshold:.3f} | weights={weights} "
        f"| profit={metrics['profit_pct']:.2f} | win-rate={metrics['win_rate']:.2f}"
    )
    logger.info(log_line)
    print(log_line)
    send_telegram_message(
        f"Tuning step:\nThreshold={threshold:.3f}\nWeights: {weights}\n"
        f"Profit={metrics['profit_pct']:.2f}\nWin-rate={metrics['win_rate']:.2f}"
    )
    return metrics["profit_pct"]


def tune_threshold():
    send_telegram_message(
        "🔄 Starter automatisk tuning af threshold og weights (Optuna)..."
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    best_threshold = study.best_params["threshold"]
    best_weights = [
        study.best_params["weight_ml"],
        study.best_params["weight_rsi"],
        study.best_params["weight_macd"],
    ]
    best_value = study.best_value
    summary = (
        f"Tuning færdig!\nBedste threshold: {best_threshold:.3f}\n"
        f"Bedste weights: {best_weights}\nProfit: {best_value:.2f}"
    )
    print("✅", summary)
    logger = get_tuning_logger()
    logger.info(summary)
    send_telegram_message("✅ " + summary)
    with open(RESULTS_PATH, "w") as f:
        f.write(
            f"Best threshold: {best_threshold:.3f}\n"
            f"Best weights: {best_weights}\n"
            f"Profit: {best_value:.2f}\n"
        )
    # --- Gem snapshot som JSON ---
    snapshot = {
        "threshold": float(best_threshold),
        "weights": [float(w) for w in best_weights],
        "timestamp": datetime.now().isoformat(),
    }
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[INFO] Snapshot gemt: {SNAPSHOT_PATH}")
    send_telegram_message(
        f"💾 Ensemble-snapshot gemt: {SNAPSHOT_PATH} \nThreshold: {best_threshold:.3f}\nWeights: {best_weights}"
    )
    return best_threshold, best_weights


if __name__ == "__main__":
    tune_threshold()
