import asyncio
import json
import time
from typing import AsyncIterator, List

import websockets

from data.schemas import Bar

BINANCE_WS = "wss://stream.binance.com:9443/ws"


def _stream_name(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"


async def subscribe(symbols: List[str], interval: str = "1m") -> AsyncIterator[Bar]:
    # 1 stream pr symbol for enkelhed/stabilitet
    async def _one(sym: str):
        stream = _stream_name(sym, interval)
        url = f"{BINANCE_WS}"
        async with websockets.connect(url, ping_interval=20, close_timeout=1) as ws:
            await ws.send(json.dumps({"method": "SUBSCRIBE", "params": [stream], "id": 1}))
            async for raw in ws:
                msg = json.loads(raw)
                k = msg.get("k")
                # kline
                if not k:
                    continue
                yield Bar(
                    ts=int(k["T"]),
                    symbol=sym.upper(),
                    venue="binance",
                    interval=interval,
                    open=float(k["o"]),
                    high=float(k["h"]),
                    low=float(k["l"]),
                    close=float(k["c"]),
                    volume=float(k["v"]),
                    is_final=bool(k["x"]),
                )

    # fan-in: merge streams
    queues = {s: asyncio.Queue(maxsize=1000) for s in symbols}

    async def _pump(sym):
        async for bar in _one(sym):
            try:
                queues[sym].put_nowait(bar)
            except:
                pass

    for s in symbols:
        asyncio.create_task(_pump(s))
    while True:
        for s, q in queues.items():
            try:
                yield q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
