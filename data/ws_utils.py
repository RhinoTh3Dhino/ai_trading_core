import asyncio, json, time
from statistics import quantiles

class P99Tracker:
    def __init__(self, maxlen: int = 5000):
        self._lat = []
        self._maxlen = maxlen
    def add(self, v: float):
        self._lat.append(v); self._lat = self._lat[-self._maxlen:]
    def p99(self) -> float:
        if len(self._lat) < 10: return 0.0
        return quantiles(self._lat, n=100)[-1]

async def heartbeat(ws, interval_s: int = 15):
    while True:
        try:
            await ws.send(json.dumps({"op": "ping", "t": int(time.time()*1000)}))
        except Exception: pass
        await asyncio.sleep(interval_s)
