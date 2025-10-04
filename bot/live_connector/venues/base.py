## bot/live_connector/venues/base.py


"""
Base connector skelet til venue-klienter.
Beholder ansvar for:
- Symbol-normalisering (venue->intern)
- Livscyklus hooks (start/stop)
- Placeholder metrics-hooks (no-op indtil rigtige metrics kobles på)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional


class BaseConnector:
    venue: str = "base"

    def __init__(
        self,
        cfg: Dict[str, Any],
        symbol_map: Dict[str, Dict[str, str]],
        on_kline: Optional[Callable[[dict], None]] = None,
        now_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self.cfg = cfg
        self.symbol_map = symbol_map
        self.on_kline = on_kline or (lambda e: None)
        self.now_ms = now_ms or (lambda: 0)
        self._stopped = False

    # ---- public API ---------------------------------------------------------
    async def run(self) -> None:
        """Kør connectoren (WS/REST loop). Skal implementeres i subclass."""
        raise NotImplementedError("Implementér i subclass")

    def shutdown(self) -> None:
        self._stopped = True

    # ---- helpers ------------------------------------------------------------
    def to_internal_symbol(self, venue_symbol: str) -> str:
        """Map fx 'BTC-USDT' (OKX) -> 'BTCUSDT' (intern)."""
        for internal, per_venue in self.symbol_map.items():
            if per_venue.get(self.venue) == venue_symbol:
                return internal
        return venue_symbol  # fallback

    # ---- metrics placeholders (no-op) --------------------------------------
    def inc_bars(self, symbol: str) -> None:
        pass

    def observe_close_lag_ms(self, symbol: str, lag_ms: int) -> None:
        pass

    def inc_reconnect(self) -> None:
        pass
