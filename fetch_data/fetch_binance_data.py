import argparse
import glob
import os
import pandas as pd
import sys
from datetime import datetime, timedelta

# Sikrer import virker uanset hvorfra scriptet køres:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.telegram_utils import send_message

from binance.client import Client
import ta  # pip install ta

def add_technical_indicators(df):
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["regime"] = 0
    df.loc[df["ema_50"] > df["ema_200"], "regime"] = 0  # bull
    df.loc[df["ema_50"] < df["ema_200"], "regime"] = 1  # bear
    return df

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
    filename = f"{symbol}_{interval}_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    outpath = os.path.join(outdir, filename)
    try:
        df = fetch_binance_ohlcv(symbol, interval, lookback_days, rolling_window)
        if set(['timestamp', 'open', 'high', 'low', 'close', 'volume']).issubset(df.columns):
            df.to_csv(outpath, index=False)
            msg = f"✅ Data hentet: {symbol} {interval}, {len(df)} rækker, fil: {filename}"
            print(msg)
            send_message(msg)
            with open("BotStatus.md", "a", encoding="utf-8") as logf:
                logf.write(f"[{now}] {msg}\n")
            return df
        else:
            raise Exception("En eller flere nødvendige kolonner mangler i hentet data.")
    except Exception as e:
        files = sorted(glob.glob(os.path.join(outdir, f"{symbol}_{interval}_*.csv")), key=os.path.getmtime)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--rolling_window", type=int, default=None, help="Antal seneste bars at bruge (valgfrit)")
    args = parser.parse_args()
    df = fetch_and_save(
        args.symbol, args.interval, args.outdir, args.lookback, args.rolling_window
    )
