import contextlib

import websockets


class WSContext:
    def __init__(self, ws):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, exc_type, exc, tb):
        with contextlib.suppress(Exception):
            await self.ws.close()


class WSClient:
    async def connect(self, url: str):
        ws = await websockets.connect(url, ping_interval=15, ping_timeout=15)
        return WSContext(ws)
