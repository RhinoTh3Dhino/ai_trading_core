# data/live_feed.py
from __future__ import annotations

from typing import Any, Dict, Optional

import ccxt
import pandas as pd

from utils.artifacts import rotate_partition  # Fase 4: skriv til part-*.parquet


# ---------------------------
# Exchange klient (ccxt)
# ---------------------------
def make_exchange(exchange_id: str, api_key: Optional[str] = None, secret: Optional[str] = None):
    """
    Lav en ccxt-exchange klient med rate limit.
    Eksempel: make_exchange("binance")
    """
    klass = getattr(ccxt, exchange_id)
    args: Dict[str, Any] = {"enableRateLimit": True, "options": {}}
    if api_key and secret:
        args["apiKey"] = api_key
        args["secret"] = secret
    ex = klass(args)
    return ex


# ---------------------------
# OHLCV fetch → DataFrame
# ---------------------------
def fetch_ohlcv_df(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
) -> pd.DataFrame:
    """
    Hent OHLCV via ccxt og returnér DataFrame i UTC med kolonner:
    ['ts','open','high','low','close','volume','timestamp']
    - ts = pandas datetime64[ns, UTC] (bruges til sort/dedup i persist)
    """
    ex = make_exchange(exchange_id, api_key, secret)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    # CCXT: [ms, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # → UTC-aware timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Fase 4: standardiser 'ts' kolonnen (dedup/sort nøgle)
    df["ts"] = df["timestamp"]
    # Kolonneorden (ts først)
    df = df[["ts", "open", "high", "low", "close", "volume", "timestamp"]]
    return df


# ---------------------------
# Persistens (Fase 4)
# ---------------------------
def persist_ohlcv_batch(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
) -> str:
    """
    Hent en batch OHLCV og skriv den som én part-fil for (symbol,timeframe) i dagens partition.
    Returnerer den fulde sti til den skrevne part-fil.
    - Vi bruger symbol_key = symbol.replace('/', '') til mappestrukturen (BTC/USDT → BTCUSDT).
    """
    df = fetch_ohlcv_df(exchange_id, symbol, timeframe, limit, api_key, secret)
    if df.empty:
        raise ValueError("persist_ohlcv_batch: tomt DataFrame fra fetch_ohlcv_df")

    # Brug fil/sti-venlig symbolnøgle i persistenslaget (samme konvention som i resten af repoet)
    symbol_key = symbol.replace("/", "")
    out_path = rotate_partition(symbol_key, timeframe, df)
    return str(out_path)


# ---------------------------
# Lille CLI til lokalkørsel
# ---------------------------
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Fetch OHLCV og (valgfrit) persistér som Parquet part.")
    p.add_argument("--exchange", required=True, help="ccxt exchange id, fx 'binance'")
    p.add_argument("--symbol", required=True, help="fx 'BTC/USDT'")
    p.add_argument("--timeframe", required=True, help="fx '1m', '5m', '1h'")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--persist", action="store_true", help="Skriv til dags-partition (Fase 4).")
    return p.parse_args()


def main() -> int:
    ns = _parse_args()
    if ns.persist:
        path = persist_ohlcv_batch(ns.exchange, ns.symbol, ns.timeframe, ns.limit)
        print(f"✅ Skrev part: {path}")
    else:
        df = fetch_ohlcv_df(ns.exchange, ns.symbol, ns.timeframe, ns.limit)
        print(df.tail(5).to_string(index=False))
        print(f"✅ Rækker hentet: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "make_exchange",
    "fetch_ohlcv_df",
    "persist_ohlcv_batch",
]
