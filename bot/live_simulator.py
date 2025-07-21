# bot/live_simulator.py


import pandas as pd
import numpy as np
import glob
from datetime import datetime
import argparse


from utils.log_utils import log_device_status
from utils.telegram_utils import send_message, send_strategy_metrics
from utils.monitoring_utils import (
    calculate_live_metrics,
    check_drawdown_alert,
    check_winrate_alert,
    check_profit_alert,
)
from utils.file_utils import save_with_metadata
from backtest.backtest import run_backtest
from bot.engine import load_ml_model, load_trained_feature_list, reconcile_features

# === Monitoring-parametre og thresholds fra central config ===
try:
    from config.monitoring_config import (
        ALARM_THRESHOLDS,
        ALERT_ON_DRAWNDOWN,
        ALERT_ON_WINRATE,
        ALERT_ON_PROFIT,
        ENABLE_MONITORING,
        LIVE_SIM_FEATURES_PATH,
        LIVE_SIM_INITIAL_BALANCE,
        LIVE_SIM_NROWS,
        LIVE_SIM_CHAT_ID,
        MODEL_TYPE,
        LIVE_SIM_SYMBOL,
        LIVE_SIM_TIMEFRAME,
        LIVE_SIM_FEATURES_DIR,
    )
except ImportError:
    ALARM_THRESHOLDS = {"drawdown": -20, "winrate": 20, "profit": -10}
    ALERT_ON_DRAWNDOWN = True
    ALERT_ON_WINRATE = True
    ALERT_ON_PROFIT = True
    ENABLE_MONITORING = True
    LIVE_SIM_FEATURES_PATH = "outputs/feature_data/live_features.csv"
    LIVE_SIM_INITIAL_BALANCE = 1000
    LIVE_SIM_NROWS = 300
    LIVE_SIM_CHAT_ID = None
    MODEL_TYPE = "ML"
    LIVE_SIM_SYMBOL = "btc"
    LIVE_SIM_TIMEFRAME = "1h"
    LIVE_SIM_FEATURES_DIR = "outputs/feature_data"

def find_latest_feature_csv(symbol="btc", timeframe="1h", feature_dir="outputs/feature_data"):
    """Finder den nyeste feature-CSV for valgt symbol/timeframe."""
    pattern = f"{feature_dir}/{symbol.lower()}_{timeframe}_features*.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Ingen feature-CSV fundet for {symbol} {timeframe} i {feature_dir}!")
    latest_file = max(files, key=os.path.getmtime)
    print(f"[INFO] Seneste feature-CSV: {latest_file}")
    return latest_file

def load_latest_features(
    symbol=LIVE_SIM_SYMBOL,
    timeframe=LIVE_SIM_TIMEFRAME,
    feature_dir=LIVE_SIM_FEATURES_DIR,
    n_rows=LIVE_SIM_NROWS
):
    """Loader de seneste rækker fra NYESTE feature-fil."""
    features_path = find_latest_feature_csv(symbol, timeframe, feature_dir)
    df = pd.read_csv(features_path)
    df = df.tail(n_rows).reset_index(drop=True)
    print(f"[INFO] Indlæst {len(df)} rækker fra {features_path}")
    return df

def load_signals(df, model_type=MODEL_TYPE):
    """Kør inference med trænet model og returnér signaler (long/short/hold)."""
    try:
        if model_type == "ML":
            ml_model, ml_features = load_ml_model()
            print(f"[DEBUG] ml_model type: {type(ml_model)}")
            if ml_model is None or ml_features is None:
                print("[FEJL] Ingen ML-model/feature-liste fundet – fallback til random signaler!")
                df["signal"] = np.random.choice([1, -1], size=len(df))
                return df
            if not hasattr(ml_model, "predict"):
                print("[FEJL] ML-model har ikke predict-metode! Fallback til random signaler.")
                df["signal"] = np.random.choice([1, -1], size=len(df))
                return df
            X = reconcile_features(df, ml_features)
            try:
                df["signal"] = ml_model.predict(X)
            except Exception as ex:
                print(f"[FEJL] Ved model.predict: {ex} – fallback til random signaler!")
                df["signal"] = np.random.choice([1, -1], size=len(df))
            print(f"[INFO] ML-signaler genereret for {len(df)} rækker.")
        else:
            df["signal"] = np.random.choice([1, -1], size=len(df))
            print(f"[INFO] Random signaler genereret (model_type={model_type}).")
    except Exception as e:
        print(f"[EXCEPTION] under signalgenerering: {e}")
        df["signal"] = np.random.choice([1, -1], size=len(df))
    return df

def metrics_fallback():
    """Returnér fallback metrics dict hvis alt fejler."""
    return {
        "profit_pct": None,
        "win_rate": None,
        "drawdown_pct": None,
        "num_trades": 0,
        "profit_factor": None,
        "sharpe": None
    }

def main(features_path=None, n_rows=LIVE_SIM_NROWS, symbol=LIVE_SIM_SYMBOL, timeframe=LIVE_SIM_TIMEFRAME):
    log_device_status(context="live_simulator", print_console=True, telegram_func=send_message)

    # 1. Indlæs feature-data (enten valgt fil eller standard/nyeste)
    try:
        if features_path:
            df = pd.read_csv(features_path)
            df = df.tail(n_rows).reset_index(drop=True)
            print(f"[INFO] Indlæst {len(df)} rækker fra {features_path}")
        else:
            df = load_latest_features(symbol=symbol, timeframe=timeframe, n_rows=n_rows)
    except Exception as e:
        msg = f"[FEJL] Live-simulering: Kunne ikke loade features ({e})"
        print(msg)
        send_message("❌ Live-simulering FEJL: Kunne ikke loade features ({})".format(e))
        return metrics_fallback()  # Returnér fallback-metrics ved fejl

    # 2. Generér signaler
    try:
        df = load_signals(df)
    except Exception as e:
        print(f"[EXCEPTION] i load_signals: {e}")
        send_message(f"❌ FEJL ved signalgenerering: {e}")
        df["signal"] = np.random.choice([1, -1], size=len(df))

    # 3. Kør simulerede handler (paper trading)
    try:
        trades_df, balance_df = run_backtest(df, signals=df["signal"].values, initial_balance=LIVE_SIM_INITIAL_BALANCE)
        print("TRADES DF:\n", trades_df)
        print("BALANCE DF:\n", balance_df)
    except Exception as e:
        msg = f"[FEJL] i run_backtest: {e}"
        print(msg)
        send_message(f"❌ FEJL i run_backtest: {e}")
        return metrics_fallback()

    # 4. Udregn metrics og performance
    try:
        metrics = calculate_live_metrics(trades_df, balance_df, initial_balance=LIVE_SIM_INITIAL_BALANCE)
        print("Live-metrics:", metrics)
        save_with_metadata(trades_df, "outputs/live_trades.csv")
        save_with_metadata(balance_df, "outputs/live_balance.csv")
    except Exception as e:
        msg = f"[FEJL] i metricsberegning/gem: {e}"
        print(msg)
        send_message(f"❌ FEJL i metricsberegning/gem: {e}")
        metrics = metrics_fallback()

    # 5. Send daglig status til Telegram
    try:
        send_strategy_metrics(metrics, chat_id=LIVE_SIM_CHAT_ID)
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke sende strategi-metrics: {e}")

    # 6. Live-monitorering & alarmer (alt styres via config)
    if ENABLE_MONITORING:
        try:
            alerts = []
            if ALERT_ON_DRAWNDOWN and check_drawdown_alert(metrics, threshold=ALARM_THRESHOLDS.get("drawdown")):
                alerts.append(
                    f"🚨 ALARM: Drawdown er {metrics.get('drawdown_pct', 0):.2f}% (grænse: {ALARM_THRESHOLDS.get('drawdown', '-'):.0f}%)"
                )
            if ALERT_ON_WINRATE and check_winrate_alert(metrics, threshold=ALARM_THRESHOLDS.get("winrate")):
                alerts.append(
                    f"⚠️ ADVARSEL: Win-rate er {metrics.get('win_rate', 0):.1f}% (grænse: {ALARM_THRESHOLDS.get('winrate', '-'):.0f}%)"
                )
            if ALERT_ON_PROFIT and check_profit_alert(metrics, threshold=ALARM_THRESHOLDS.get("profit")):
                alerts.append(
                    f"‼️ ADVARSEL: Profit er {metrics.get('profit_pct', 0):.2f}% (grænse: {ALARM_THRESHOLDS.get('profit', '-'):.0f}%)"
                )
            for alert in alerts:
                send_message(alert, chat_id=LIVE_SIM_CHAT_ID)
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke sende alarmer: {e}")

    print("Live-simulering færdig!")
    return metrics  # VIGTIGT: Returnér altid metrics (også ved fejl!)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None, help="Sti til feature-fil (CSV)")
    parser.add_argument("--n_rows", type=int, default=LIVE_SIM_NROWS, help="Antal rækker at tage med")
    parser.add_argument("--symbol", type=str, default=LIVE_SIM_SYMBOL, help="Symbol (fx btcusdt)")
    parser.add_argument("--timeframe", type=str, default=LIVE_SIM_TIMEFRAME, help="Timeframe (fx 1h)")
    args = parser.parse_args()
    main(features_path=args.features, n_rows=args.n_rows, symbol=args.symbol, timeframe=args.timeframe)
