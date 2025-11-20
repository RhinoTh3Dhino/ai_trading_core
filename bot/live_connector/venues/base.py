import asyncio
import logging
from typing import Any, Callable, Dict

log = logging.getLogger(__name__)


class BaseConnector:
    venue: str = "base"

    def __init__(self, cfg: dict, symbol_map: dict, on_kline: Callable[[dict], None], ws_client):
        self.cfg = cfg
        self.symbol_map = symbol_map
        self.on_kline = on_kline
        self.ws_client = ws_client
        self._stop = False

    async def run(self):
        backoff = 1
        while not self._stop:
            try:
                async with await self.ws_client.connect(self.cfg["ws"]["url"]) as ws:
                    await self._subscribe(ws)
                    backoff = 1
                    async for msg in self._read_loop(ws):
                        k = self._parse_kline(msg)
                        if k:
                            self.on_kline(k)
            except Exception as e:
                log.warning("Reconnect %s: %s", self.venue, e)
                await asyncio.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)

    async def _subscribe(self, ws): ...
    async def _read_loop(self, ws): ...
    def _parse_kline(self, msg: Dict[str, Any]): ...
