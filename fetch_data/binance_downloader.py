# fetch_data/binance_downloader.py

import pandas as pd
from datetime import datetime
from time import sleep



from config.config import COINS, TIMEFRAMES

# Prøv at importere Binance-klienten
try:
    from binance.client import Client
except ImportError:
    print("Installér først python-binance:\n  pip install python-binance")
    exit(1)

# Sæt API-nøgler her (eller brug miljøvariabler)
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)

def download_binance_ohlcv(symbol, interval, start_str, end_str, out_csv):
    print(f"Henter {symbol} {interval} fra {start_str} til {end_str} ...")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset', 'n_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Behold kun relevante kolonner og gem som float (med punktum!)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Gem CSV med ; og punktum som decimal (dansk/international friendly)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, sep=";", decimal=".")
    print(f"✅ Gemte {len(df)} rækker til {out_csv}")

if __name__ == "__main__":
    # Indstil hvor langt tilbage du vil hente (fx 2 år)
    START_DATE = "2 years ago UTC"
    END_DATE = "now UTC"

    for symbol in COINS:
        for tf in TIMEFRAMES:
            # Binance interval format (1h, 4h, 1d ...)
            tf_binance = tf if "m" in tf or "h" in tf or "d" in tf else f"{tf}"
            out_csv = f"data/{symbol}_{tf}.csv"
            try:
                download_binance_ohlcv(
                    symbol=symbol,
                    interval=tf_binance,
                    start_str=START_DATE,
                    end_str=END_DATE,
                    out_csv=out_csv
                )
                # Undgå at ramme rate-limits!
                sleep(1)
            except Exception as e:
                print(f"❌ Fejl for {symbol} {tf}: {e}")
                continue

    print("\n🚀 Download færdig! Kør nu din features_pipeline.py på de nye CSV'er.")
