import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_training import train_model
from backtest.backtest import run_backtest, calc_backtest_metrics
from visualization.plot_backtest import plot_backtest
from visualization.plot_drawdown import plot_drawdown
from utils.telegram_utils import send_telegram_photo, send_telegram_message
from utils.robust_utils import safe_run
from ensemble.majority_vote_ensemble import weighted_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

# --- Optuna-tuning support ---
try:
    from tuning.tuning_threshold import tune_threshold  # Denne funktion skal findes i tuner/tuning_threshold.py
except ImportError:
    tune_threshold = None

DATA_PATH = "outputs/feature_data/btc_1h_features_v_test_20250610.csv"
SYMBOL = "BTC"
GRAPH_DIR = "graphs/"

# --- Ensemble voting mode ---
USE_WEIGHTED = True   # True = Weighted Voting, False = Majority Voting
WEIGHTS = [1.0, 0.7, 0.4]  # ML, RSI, MACD (bruges kun hvis weighted)

# --- (Ekstra) Dynamisk threshold fra tuning ---
# Standard-threshold til strategi (hvis det skal bruges)
DEFAULT_THRESHOLD = 0.7

def main(threshold=DEFAULT_THRESHOLD):
    print("🔄 Indlæser features:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data indlæst ({len(df)} rækker)")
    print("Kolonner:", list(df.columns))

    # ML-model
    print("🔄 Træner eller indlæser ML-model ...")
    model, model_path, feature_cols = train_model(df)
    print(f"✅ ML-model klar: {model_path}")
    X_pred = df[feature_cols]
    # Her kan du indsætte thresholding hvis din ML-model outputter probabiliteter
    ml_raw = model.predict(X_pred)
    # Hvis model.predict returnerer probabiliteter:
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_pred)[:, 1]
        ml_signals = (probas > threshold).astype(int)
    else:
        ml_signals = ml_raw  # Tilpas hvis din model allerede outputter 0/1 signaler

    # Indikator-strategier
    print("🔄 Genererer strategi-signaler ...")
    rsi_signals = rsi_rule_based_signals(df, low=30, high=70)
    macd_signals = macd_cross_signals(df)
    print(f"Signal distribution ML/RSI/MACD:",
          pd.Series(ml_signals).value_counts().to_dict(),
          pd.Series(rsi_signals).value_counts().to_dict(),
          pd.Series(macd_signals).value_counts().to_dict())

    # Ensemble voting
    if USE_WEIGHTED:
        print(f"➡️  Bruger vægtet voting med weights: {WEIGHTS}")
        ensemble_signals = weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, weights=WEIGHTS)
    else:
        print("➡️  Bruger klassisk majority voting")
        ensemble_signals = weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals)  # weights=None

    df["signal"] = ensemble_signals

    # Backtest
    print("🔄 Kører backtest ...")
    trades_df, balance_df = run_backtest(df, signals=ensemble_signals)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    print("Backtest-metrics:", metrics)

    # Grafer
    print("🔄 Genererer grafer ...")
    plot_path = plot_backtest(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)
    drawdown_path = plot_drawdown(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)

    # Telegram
    print("🔄 Sender grafer til Telegram ...")
    send_telegram_message(
        f"✅ Backtest for {SYMBOL} afsluttet!\n"
        f"Mode: {'Weighted' if USE_WEIGHTED else 'Majority'} voting\n"
        f"Weights: {WEIGHTS if USE_WEIGHTED else '[1,1,1]'}\n"
        f"Threshold: {threshold}\n"
        f"Profit: {metrics['profit_pct']}% | Win-rate: {metrics['win_rate']*100:.1f}% | Trades: {metrics['num_trades']}"
    )
    send_telegram_photo(plot_path, caption=f"📈 Balanceudvikling for {SYMBOL}")
    send_telegram_photo(drawdown_path, caption=f"📉 Drawdown for {SYMBOL}")

    print("🎉 Hele flowet er nu automatisk!")

if __name__ == "__main__":
    # Mulighed 1: Kør tuning før main (via argument eller konfig)
    if "--tune" in sys.argv and tune_threshold:
        send_telegram_message("🔧 Starter automatisk tuning af threshold...")
        best_threshold = tune_threshold()
        send_telegram_message(f"🏆 Bedste fundne threshold: {best_threshold:.3f} – genstarter backtest med ny værdi.")
        safe_run(lambda: main(threshold=best_threshold))
    else:
        safe_run(main)
