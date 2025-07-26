import ccxt
import pandas as pd
from datetime import datetime

from utils.project_path import PROJECT_ROOT
# ---- Relativt import-trick: Sikrer at 'utils' kan importeres uanset hvorfra scriptet køres ----
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.robust_utils import safe_run  # ← Tilføjet robusthed!

def hent_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000, start=None, slut=None, filnavn=None):
    binance = ccxt.binance()
    since = None
    if start:
        since = int(pd.Timestamp(start).timestamp() * 1000)

    # Hent data fra Binance
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)

    # Foreslået filnavn hvis ikke angivet
    if not filnavn:
# AUTO PATH CONVERTED
        filnavn = fPROJECT_ROOT / "data" / "{symbol.replace("/', '')}_{timeframe}.csv"
    df.to_csv(filnavn, sep=";", decimal=",")
    print(f"✅ Data gemt: {filnavn}")
    return df

def main():
    # Eksempel: Hent de seneste 1000 1h-bars for BTC/USDT
    hent_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000)

if __name__ == "__main__":
    safe_run(main)