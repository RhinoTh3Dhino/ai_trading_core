# fetch_data/fetch_binance_data.py

import argparse
import glob

import pandas as pd

from datetime import datetime, timedelta


from utils.telegram_utils import send_message

from binance.client import Client

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1h", lookback_days=30, rolling_window=None):
    client = Client(api_key=None, api_secret=None)
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    klines = client.get_historical_klines(
        symbol, interval,
        start.strftime('%d %b %Y %H:%M:%S'),
        end.strftime('%d %b %Y %H:%M:%S')
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    if rolling_window:
        df = df.iloc[-rolling_window:]  # Kun de seneste N rækker
        print(f"ℹ️ Bruger kun de seneste {rolling_window} bars fra Binance-data.")
    return df

def fetch_and_save(symbol: str, interval: str, outdir: str = "data", lookback_days: int = 30, rolling_window: int = None) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    now = datetime.now()
    filename = f"{symbol.upper()}_{interval}_{lookback_days}d.csv"
    outpath = os.path.join(outdir, filename)
    try:
        df = fetch_binance_ohlcv(symbol, interval, lookback_days, rolling_window)
        # Konverter kolonner til numerisk datatype (undtagen timestamp)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop NaN rækker hvis de findes
        before = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        after = len(df)
        if before != after:
            print(f"ℹ️ {before-after} rækker droppet pga. NaN i OHLCV.")
        df.to_csv(outpath, index=False)
        msg = f"✅ Rå OHLCV-data gemt: {symbol} {interval}, {len(df)} rækker, fil: {filename}"
        print(msg)
        send_message(msg)
        with open("BotStatus.md", "a", encoding="utf-8") as logf:
            logf.write(f"[{now}] {msg}\n")
        return df
    except Exception as e:
        files = sorted(glob.glob(os.path.join(outdir, f"{symbol.upper()}_{interval}_*.csv")), key=os.path.getmtime)
        if files:
            fallback = files[-1]
            df = pd.read_csv(fallback)
            msg = f"⚠️ Fetch FEJLEDE, bruger fallback: {fallback}"
            print(msg)
            send_message(msg)
            with open("BotStatus.md", "a", encoding="utf-8") as logf:
                logf.write(f"[{now}] {msg}\n")
            return df
        else:
            msg = f"❌ Fetch FEJLEDE og ingen fallback-fil fundet: {e}"
            print(msg)
            send_message(msg)
            with open("BotStatus.md", "a", encoding="utf-8") as logf:
                logf.write(f"[{now}] {msg}\n")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hent og gem rå OHLCV-data fra Binance til CSV.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol, fx BTCUSDT, ETHUSDT osv.")
    parser.add_argument("--interval", type=str, default="1h", help="Tidsinterval, fx 1h, 4h, 1d")
    parser.add_argument("--outdir", type=str, default="data", help="Output-mappe for CSV")
    parser.add_argument("--lookback", type=int, default=None, help="Antal dage tilbage at hente (kan også hedde --days)")
    parser.add_argument("--days", type=int, default=None, help="Alias for --lookback")
    parser.add_argument("--rolling_window", type=int, default=None, help="Antal seneste bars at bruge (valgfrit)")
    args = parser.parse_args()

    # Håndter alias for --days
    lookback_days = args.lookback if args.lookback is not None else args.days if args.days is not None else 30

    df = fetch_and_save(
        args.symbol,
        args.interval,
        args.outdir,
        lookback_days,
        args.rolling_window
    )
