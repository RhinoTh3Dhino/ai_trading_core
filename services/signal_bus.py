# services/signal_bus.py
import asyncio
_q = asyncio.Queue(maxsize=10000)
async def publish(msg): await _q.put(msg)
async def subscribe():
    while True:
        yield await _q.get()
