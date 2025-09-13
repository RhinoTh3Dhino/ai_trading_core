from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

Venue = Literal["binance", "bybit", "okx", "kraken"]

class Bar(BaseModel):
    ts: int                   # epoch ms (bar close)
    symbol: str               # e.g. BTCUSDT
    venue: Venue
    interval: str             # "1m"
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_final: bool = True

class FeedMetric(BaseModel):
    ts: int
    venue: Venue
    symbol: str
    latency_ms: int
    seq_ok: bool
    gap_detected: bool = False
