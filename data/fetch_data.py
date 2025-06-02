import ccxt
import pandas as pd
from datetime import datetime

def hent_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000, start=None, slut=None, filnavn=None):
    binance = ccxt.binance()
    since = None
    if start:
        # Konverter til ms-timestamp (Binance kræver dette)
        since = int(pd.Timestamp(start).timestamp() * 1000)

    # Hent data fra Binance
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)

    # Foreslået filnavn hvis ikke angivet
    if not filnavn:
        filnavn = f"data/{symbol.replace('/', '')}_{timeframe}.csv"
    df.to_csv(filnavn, sep=";", decimal=",")
    print(f"✅ Data gemt: {filnavn}")
    return df

if __name__ == "__main__":
    # Eksempel: Hent de seneste 1000 1h-bars for BTC/USDT
    hent_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000)
