# data/live_feed.py
from __future__ import annotations

from typing import Optional

import ccxt
import pandas as pd


def make_exchange(
    exchange_id: str, api_key: Optional[str] = None, secret: Optional[str] = None
):
    klass = getattr(ccxt, exchange_id)
    args = {"enableRateLimit": True}
    if api_key and secret:
        args["apiKey"] = api_key
        args["secret"] = secret
    ex = klass(args)
    return ex


def fetch_ohlcv_df(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
) -> pd.DataFrame:
    ex = make_exchange(exchange_id, api_key, secret)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    # CCXT: [ms, open, high, low, close, volume]
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
