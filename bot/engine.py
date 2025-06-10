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

from ensemble.majority_vote_ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

DATA_PATH = "outputs/feature_data/btc_1h_features_v_test_20250610.csv"
SYMBOL = "BTC"
GRAPH_DIR = "graphs/"

def main():
    print("🔄 Indlæser features:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data indlæst ({len(df)} rækker)")
    print("Kolonner:", list(df.columns))
    
    # Hvis du vil køre UDEN ML-model (kun dummy til test)
    # np.random.seed(42)
    # ml_signals = np.random.choice([1, 0, -1], size=len(df))
    # Ellers brug model:
    print("🔄 Træner eller indlæser ML-model ...")
    model, model_path, feature_cols = train_model(df)
    print(f"✅ ML-model klar: {model_path}")
    X_pred = df[feature_cols]
    ml_signals = model.predict(X_pred)

    # Strategi-signaler
    print("🔄 Genererer strategi-signaler ...")
    rsi_signals = rsi_rule_based_signals(df, low=30, high=70)
    macd_signals = macd_cross_signals(df)
    print(f"Signal distribution ML/RSI/MACD:",
          pd.Series(ml_signals).value_counts().to_dict(),
          pd.Series(rsi_signals).value_counts().to_dict(),
          pd.Series(macd_signals).value_counts().to_dict())
    
    # Ensemble
    ensemble_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals)
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
    send_telegram_message(f"✅ Backtest for {SYMBOL} afsluttet!\nSe balance og drawdown-graf.\n"
                          f"Profit: {metrics['profit_pct']}% | Win-rate: {metrics['win_rate']*100:.1f}% | Trades: {metrics['num_trades']}")
    send_telegram_photo(plot_path, caption=f"📈 Balanceudvikling for {SYMBOL}")
    send_telegram_photo(drawdown_path, caption=f"📉 Drawdown for {SYMBOL}")

    print("🎉 Hele flowet er nu automatisk!")

if __name__ == "__main__":
    safe_run(main)
