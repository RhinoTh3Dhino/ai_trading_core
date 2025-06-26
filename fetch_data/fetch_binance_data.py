import argparse
import glob
import os
import pandas as pd
from datetime import datetime
from utils.telegram_utils import send_message
import datetime
from binance.client import Client
import ta  # pip install ta
import sys
import subprocess

def add_technical_indicators(df):
    df = df.copy()
    # Konverter alle relevante kolonner til float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # RSI (14)
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    # EMA
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    # ATR (volatilitet)
    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    # Regime (bull hvis ema_50 > ema_200, bear hvis omvendt)
    df["regime"] = 0
    df.loc[df["ema_50"] > df["ema_200"], "regime"] = 0  # bull
    df.loc[df["ema_50"] < df["ema_200"], "regime"] = 1  # bear

    return df

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1h", lookback_days=30, save_path=None):
    client = Client(api_key=None, api_secret=None)
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=lookback_days)
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
    df = add_technical_indicators(df)
    if save_path:
        df.to_csv(save_path, index=False)
    return df

def fetch_data(symbol: str, interval: str, outdir: str = "data") -> pd.DataFrame:
    """
    Henter data fra Binance (mockup), gemmer som CSV, logger til Telegram og BotStatus.md.
    Fallback: Hvis fetch fejler, brug seneste gyldige fil.
    """
    # TODO: Erstat dette mockup med rigtig Binance-fetch
    try:
        # Simuleret data (erstat med rigtig fetch)
        now = datetime.now()
        df = pd.DataFrame({
            "timestamp": pd.date_range(now, periods=100, freq="H"),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100
        })
        filename = f"{symbol}_{interval}_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        outpath = os.path.join(outdir, filename)
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(outpath, index=False)
        msg = f"✅ Data hentet: {symbol} {interval}, {len(df)} rækker, fil: {filename}"
        print(msg)
        send_message(msg)
        # Log evt. til BotStatus.md her
        return df
    except Exception as e:
        # Fallback: find seneste fil
        files = sorted(glob.glob(os.path.join(outdir, f"{symbol}_{interval}_*.csv")), key=os.path.getmtime)
        if files:
            fallback = files[-1]
            df = pd.read_csv(fallback)
            msg = f"⚠️ Fetch FEJLEDE, bruger fallback: {fallback}"
            print(msg)
            send_message(msg)
            return df
        else:
            msg = f"❌ Fetch FEJLEDE og ingen fallback-fil fundet: {e}"
            print(msg)
            send_message(msg)
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()
    fetch_data(args.symbol, args.interval, args.outdir)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    save_path = f"outputs/feature_data/btc_1h_features_{date_str}.csv"
    fetch_binance_ohlcv(save_path=save_path)
    print("✅ Data gemt til:", save_path)

    # BONUS: Kald feature engineering automatisk efter fetch!
    # Så hele pipeline kan køres direkte fra dette script hvis ønsket.
    # Kommentér ud hvis du bruger run_all.py til at styre hele flowet!
    # subprocess.run([sys.executable, "features/feature_engineering.py"])
