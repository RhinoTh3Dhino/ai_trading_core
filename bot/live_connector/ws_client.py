# bot/live_connector/ws_client.py
"""
Minimal WSClient-stub til dev/test.
- Ingen eksterne afhængigheder.
- Metoderne er no-ops, så connector-kode kan køre i tør-mode.
- Hvis du vil bruge “rigtig” websocket, kan du senere udvide med websockets/aiohttp.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class WSClient:
    def __init__(self, url: Optional[str] = None, headers: Optional[dict] = None, loop=None):
        self.url = url
        self.headers = headers or {}
        self.loop = loop or asyncio.get_event_loop()
        self.connected = False

    async def connect(self, url: Optional[str] = None, headers: Optional[dict] = None) -> None:
        """Simuler en forbindelse (no-op)."""
        if url:
            self.url = url
        if headers:
            self.headers.update(headers)
        self.connected = True

    async def send(self, data: Any) -> None:
        """No-op send; accepterer str/bytes/dict."""
        _ = data  # stil lint/black tilfreds

    async def send_json(self, obj: Any) -> None:
        """Alias for send() i stubben."""
        await self.send(obj)

    async def recv(self) -> Any:
        """Returnér straks en tom payload i stubben."""
        await asyncio.sleep(0)
        return None  # kunne være bytes/str i en rigtig klient

    async def close(self) -> None:
        """Luk 'forbindelsen' (no-op)."""
        self.connected = False

    # Praktisk context-manager i async brug
    async def __aenter__(self) -> "WSClient":
        await self.connect(self.url, self.headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = ["WSClient"]
