import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import logging
import pandas as pd
from datetime import datetime
from telegram import Bot

from models.model_training import train_model
from backtest.backtest import run_backtest, calc_backtest_metrics
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals
from ensemble.majority_vote_ensemble import weighted_vote_ensemble

# === Telegram Setup ===
TELEGRAM_TOKEN = "DIN_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "DIN_TELEGRAM_CHAT_ID"
bot = Bot(token=TELEGRAM_TOKEN)

# === Konstanter til tuning (tilpas hvis nødvendigt) ===
DATA_PATH = "outputs/feature_data/btc_1h_features_v_test_20250610.csv"
SYMBOL = "BTC"
WEIGHTS = [1.0, 0.7, 0.4]
USE_WEIGHTED = True

# === Filstier (alt havner i tuning/) ===
TUNER_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(TUNER_DIR, "tuning_log.txt")
RESULTS_PATH = os.path.join(TUNER_DIR, "tuning_results_threshold.txt")

# --- Sikr at tuning-mappen findes ---
if not os.path.exists(TUNER_DIR):
    os.makedirs(TUNER_DIR)

print(f"[INFO] Logfil skrives til: {LOG_PATH}")
print(f"[INFO] Resultatfil skrives til: {RESULTS_PATH}")

# === Robust logger-setup (virker ved både import og direkte kørsel) ===
def get_tuning_logger():
    logger = logging.getLogger("tuning_logger")
    logger.setLevel(logging.INFO)
    # Tilføj kun handler én gang!
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_PATH, mode='w')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def send_telegram_message(message):
    async def async_send_message():
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    try:
        asyncio.run(async_send_message())
    except Exception as e:
        logger = get_tuning_logger()
        logger.error(f"Telegram-fejl: {e}")

def objective(trial):
    logger = get_tuning_logger()
    threshold = trial.suggest_float("threshold", 0.5, 0.9)
    df = pd.read_csv(DATA_PATH)
    model, _, feature_cols = train_model(df)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df[feature_cols])[:, 1]
        ml_signals = (probas > threshold).astype(int)
    else:
        ml_signals = model.predict(df[feature_cols])
    rsi_signals = rsi_rule_based_signals(df, low=30, high=70)
    macd_signals = macd_cross_signals(df)
    if USE_WEIGHTED:
        ensemble_signals = weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, weights=WEIGHTS)
    else:
        ensemble_signals = weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals)
    df["signal"] = ensemble_signals
    trades_df, balance_df = run_backtest(df, signals=ensemble_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    log_line = f"{datetime.now()} | threshold={threshold:.3f} | profit={metrics['profit_pct']:.2f} | win-rate={metrics['win_rate']:.2f}"
    logger.info(log_line)
    print(log_line)
    send_telegram_message(
        f"Tuning step:\nThreshold={threshold:.3f}\nProfit={metrics['profit_pct']:.2f}\nWin-rate={metrics['win_rate']:.2f}"
    )
    return metrics["profit_pct"]

def tune_threshold():
    send_telegram_message("🔄 Starter automatisk tuning af threshold (Optuna)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    best_threshold = study.best_params["threshold"]
    best_value = study.best_value
    summary = f"Tuning færdig!\nBedste threshold: {best_threshold:.3f}\nProfit: {best_value:.2f}"
    print("✅", summary)  # Emoji KUN til konsol/Telegram
    logger = get_tuning_logger()
    logger.info(summary)  # Uden emoji
    send_telegram_message("✅ " + summary)
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Best threshold: {best_threshold:.3f}\nProfit: {best_value:.2f}\n")
    return best_threshold

if __name__ == "__main__":
    tune_threshold()
