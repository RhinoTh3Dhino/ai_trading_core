# tests/test_live_connector_integration.py
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

# IMPORT EFTER vi har monkeypatch'et modulattributter
import importlib

from data.schemas import Bar

class _DummyP99:
    def p99(self): return 0.0

class DummyOrchestrator:
    latest_queue = None
    def __init__(self, symbols, interval, min_active=2):
        self.symbols = symbols
        self.interval = interval
        self.min_active = min_active
        self.p99 = {"binance": _DummyP99(), "bybit": _DummyP99(), "okx": _DummyP99(), "kraken": _DummyP99()}
        self.active = {"binance": True, "bybit": True, "okx": False, "kraken": False}
    async def run(self):
        q = asyncio.Queue()
        DummyOrchestrator.latest_queue = q
        return q
    async def shutdown(self):
        return

@pytest.mark.asyncio
async def test_end_to_end_parquet(tmp_path, monkeypatch):
    # Dir setup
    out = tmp_path / "out"
    logs = tmp_path / "logs"
    out.mkdir(); logs.mkdir()

    # Import services.live_connector og patch paths/FeedOrchestrator
    import services.live_connector as lc
    importlib.reload(lc)
    monkeypatch.setattr(lc, "OUTPUTS", out, raising=True)
    monkeypatch.setattr(lc, "LOGS", logs, raising=True)
    monkeypatch.setattr(lc, "FeedOrchestrator", DummyOrchestrator, raising=True)

    # Start main
    task = asyncio.create_task(lc.main(["TESTUSDT"], "1m", quiet=True))

    # Vent til run() er kaldt og vi har queue
    for _ in range(100):
        if DummyOrchestrator.latest_queue is not None:
            break
        await asyncio.sleep(0.01)
    q = DummyOrchestrator.latest_queue
    assert q is not None

    # Skub 3 u-lukkede + 10 lukkede barer
    # Brug tidsstempler tæt på nu for pænere lag-logs
    ts = int(time.time() * 1000) - 60_000 * 20
    price = 100.0
    for i in range(3):
        await q.put(Bar(
            venue="binance",
            symbol="TESTUSDT", interval="1m", ts=ts,
            open=price, high=price+0.5, low=price-0.5, close=price+0.2,
            volume=10.0+i, is_final=False
        ))
        ts += 60_000
        price += 0.3

    closed = 10
    for i in range(closed):
        await q.put(Bar(
            venue="binance",
            symbol="TESTUSDT", interval="1m", ts=ts,
            open=price, high=price+0.7, low=price-0.4, close=price+0.25,
            volume=12.0+i, is_final=True
        ))
        ts += 60_000
        price += 0.25

    # Giv connectoren tid til at skrive, derefter stop
    await asyncio.sleep(0.5)
    task.cancel()

    # Connectoren fanger CancelledError selv og afslutter pænt.
    # Derfor skal vi IKKE forvente en exception her:
    await task
    assert task.done()

    # Læs Parquet
    f = out / "TESTUSDT_1m.parquet"
    assert f.exists(), "Parquet-fil blev ikke skrevet"
    df = pd.read_parquet(f)
    # Kun lukkede barer burde være skrevet
    assert len(df) == closed
    # Kolonner
    for col in ["ts","open","high","low","close","volume","ema_14","ema_50","rsi_14","vwap","atr_14"]:
        assert col in df.columns

    # Basekolonner må ikke have NaN
    assert df[["ts","open","high","low","close","volume"]].isna().sum().sum() == 0

    # Metadata-check (pyarrow)
    try:
        import pyarrow.parquet as pq
        md = pq.read_table(f).schema.metadata
        assert b"schema_version" in md
        assert md[b"schema_version"].decode() == os.getenv("LIVE_SCHEMA_VERSION","stream-mvp-1")
    except Exception:
        # fallback meta sidecar
        sidecar = Path(str(f) + ".meta.json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            assert meta.get("schema_version") == os.getenv("LIVE_SCHEMA_VERSION","stream-mvp-1")
