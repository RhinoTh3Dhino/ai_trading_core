# bot/multi_live_simulator.py

import sys
import os
import pandas as pd
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.log_utils import log_device_status
from utils.telegram_utils import send_message
from bot.live_simulator import main as run_live_sim

# === Konfigurer coins, timeframes og parametre her ===
MULTI_CONFIG = [
    {
        "symbol": "btcusdt",
        "timeframe": "1h",
        "features_path": "outputs/feature_data/btcusdt_1h_features_v1.3_20250718.csv",
        "n_rows": 500,
    },
    {
        "symbol": "ethusdt",
        "timeframe": "1h",
        "features_path": "outputs/feature_data/ethusdt_1h_features_latest.csv",
        "n_rows": 500,
    },
    {
        "symbol": "dogeusdt",
        "timeframe": "1h",
        "features_path": "outputs/feature_data/dogeusdt_1h_features_latest.csv",
        "n_rows": 500,
    },
    # Tilføj flere coins/timeframes her!
]

def status_text(symbol, timeframe, metrics, for_console=False):
    """
    Returnér status-linje for et symbol/timeframe.
    Emoji KUN til Telegram (for_console=False).
    """
    if not isinstance(metrics, dict) or metrics.get("profit_pct") is None:
        err = metrics.get("error", "-") if isinstance(metrics, dict) else "-"
        if for_console:
            return f"{symbol.upper()} {timeframe}: FEJL eller ingen metrics retur ({err})"
        return f"⚠️ {symbol.upper()} {timeframe}: FEJL eller ingen metrics retur ({err})"
    base = (
        f"{symbol.upper()} {timeframe} | "
        f"P/L: {metrics.get('profit_pct', '-')}% | "
        f"WR: {metrics.get('win_rate', '-')}% | "
        f"DD: {metrics.get('drawdown_pct', '-')}% | "
        f"Trades: {metrics.get('num_trades', '-')}"
    )
    if for_console:
        return base
    else:
        return f"📊 {base}"

def multi_live_simulation(configs=MULTI_CONFIG, n_rows_override=None):
    all_metrics = []
    all_status_telegram = []
    log_device_status(context="multi_live_simulator", print_console=True, telegram_func=send_message)
    for conf in configs:
        symbol = conf.get("symbol")
        timeframe = conf.get("timeframe")
        features_path = conf.get("features_path")
        n_rows = n_rows_override if n_rows_override is not None else conf.get("n_rows", 300)
        print(f"\n=== Kører live-simulering for {symbol.upper()} {timeframe} ({n_rows} rækker) ===")
        try:
            # Kald single live_simulator for hvert symbol/timeframe
            metrics = run_live_sim(
                features_path=features_path,
                n_rows=n_rows,
                symbol=symbol,
                timeframe=timeframe,
            )
            status_console = status_text(symbol, timeframe, metrics, for_console=True)
            status_telegram = status_text(symbol, timeframe, metrics, for_console=False)
            all_metrics.append((symbol, timeframe, metrics))
        except Exception as e:
            err = f"FEJL i live-simulering for {symbol.upper()} {timeframe}: {e}"
            print(err)
            status_console = f"{symbol.upper()} {timeframe}: Exception – {e}"
            status_telegram = f"⚠️ {symbol.upper()} {timeframe}: Exception – {e}"
        print(status_console)
        all_status_telegram.append(status_telegram)
    # --- Send samlet Telegram-status ---
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"📡 <b>Multi Live Paper Trading Status {date_str}</b>\n\n" + "\n".join(all_status_telegram)
    send_message(msg, parse_mode="HTML")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, default=None, help="Kommasepareret liste af coins (fx 'btcusdt,ethusdt')")
    parser.add_argument("--timeframes", type=str, default=None, help="Kommasepareret liste af timeframes (fx '1h,4h')")
    parser.add_argument("--n_rows", type=int, default=None, help="Antal rækker per symbol/timeframe (overrider default)")
    args = parser.parse_args()
    # Dynamisk override via CLI (valgfrit)
    configs = MULTI_CONFIG
    if args.coins:
        coins = [c.strip().lower() for c in args.coins.split(",")]
        configs = [c for c in configs if c["symbol"] in coins]
    if args.timeframes:
        tfs = [t.strip() for t in args.timeframes.split(",")]
        configs = [c for c in configs if c["timeframe"] in tfs]
    n_rows_override = args.n_rows if args.n_rows is not None else None
    multi_live_simulation(configs=configs, n_rows_override=n_rows_override)
