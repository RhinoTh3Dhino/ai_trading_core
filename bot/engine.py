import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime

from models.model_training import train_model
from backtest.backtest import run_backtest
from visualization.plot_backtest import plot_backtest
from visualization.plot_drawdown import plot_drawdown
from utils.telegram_utils import send_telegram_photo, send_telegram_message
from utils.robust_utils import safe_run

from models.ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals

DATA_PATH = "data/BTCUSDT_1h_features.csv"
SYMBOL = "BTC"
GRAPH_DIR = "graphs/"

def main():
    # 1. Indlæs data (features)
    print("🔄 Indlæser data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data indlæst ({df.shape[0]} rækker)")
    print(f"Kolonner før evt. omdøbning: {list(df.columns)}")

    if "datetime" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = df["datetime"]
        print("ℹ️ Tilføjede 'timestamp' kolonne ud fra 'datetime'")

    print(f"Kolonner efter evt. omdøbning: {list(df.columns)}")

    # 2. Træn eller indlæs model
    print("🔄 Træner model...")
    model, model_path, feature_cols = train_model(df)
    print(f"✅ Model klar: {model_path}")

    # 3. Generér signaler (ML) + Indikator + Ensemble Voting
    print("🔄 Genererer signaler og kører backtest...")

    X_pred = df[feature_cols]
    ml_signals = model.predict(X_pred)
    indi_signals = rsi_rule_based_signals(df, low=30, high=70)
    ensemble_signals = majority_vote_ensemble(ml_signals, indi_signals)

    trades_df, balance_df = run_backtest(df, signals=ensemble_signals)
    print("✅ Backtest gennemført")

    # 4. Gem grafer
    print("🔄 Genererer grafer...")
    plot_path = plot_backtest(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)
    drawdown_path = plot_drawdown(balance_df, symbol=SYMBOL, save_dir=GRAPH_DIR)

    # 5. Send resultater til Telegram
    print("🔄 Sender grafer til Telegram...")
    send_telegram_message(f"✅ Backtest for {SYMBOL} afsluttet!\nSe balance og drawdown-graf.")
    send_telegram_photo(plot_path, caption=f"📈 Balanceudvikling for {SYMBOL}")
    send_telegram_photo(drawdown_path, caption=f"📉 Drawdown for {SYMBOL}")

    print("🎉 Hele flowet er nu automatisk!")

if __name__ == "__main__":
    safe_run(main)
