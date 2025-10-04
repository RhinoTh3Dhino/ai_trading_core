## bot/live_connector/venues/okx.py


"""
OKX connector skelet.
Fokus i første PR:
- Parsing-hjælpere der normaliserer OKX candle1m -> internt kline-event
- Ingen netværk (WS/REST) endnu -> TODO i næste iteration
"""

from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseConnector


def parse_okx_candle_payload(msg: Dict[str, Any], to_internal: callable) -> List[dict]:
    """
    Forventer OKX public stream payload:
    {
      "arg":  {"channel":"candle1m", "instId":"BTC-USDT"},
      "data": [["timestamp","o","h","l","c","vol","volCcy","volCcyQuote","confirm"]]
    }
    Returnerer liste af normaliserede kline-events (kan være >1 ved snapshot).
    """
    if "arg" not in msg or "data" not in msg:
        return []

    inst = msg["arg"].get("instId", "")
    internal_symbol = to_internal(inst)

    out: List[dict] = []
    for row in msg["data"]:
        # OKX tidsstempel er normalt ms (str); confirm er "0"/"1"
        ts_ms = int(row[0])
        o, h, l, c = map(float, row[1:5])
        v = float(row[5])
        is_final = bool(int(row[8])) if len(row) > 8 else False

        evt = {
            "venue": "okx",
            "symbol": internal_symbol,
            "tf": "1m",
            "open_time": ts_ms,
            "close_time": ts_ms + 60_000 - 1,
            "o": o,
            "h": h,
            "l": l,
            "c": c,
            "v": v,
            "is_final": is_final,
            "event_ts_ms": ts_ms,  # midlertidigt; kan sættes til 'now' ved real-time
        }
        out.append(evt)
    return out


class OKXConnector(BaseConnector):
    venue = "okx"

    async def run(self) -> None:
        """
        TODO (næste iteration):
        - Etabler WS: wss://ws.okx.com:8443/ws/v5/public
        - Send subscribe: {"op":"subscribe","args":[{"channel":"candle1m","instId":"BTC-USDT"}, ...]}
        - for hver besked: parse_okx_candle_payload(...) og self.on_kline(evt)
        - metrics-hooks (inc_bars, observe_close_lag_ms, inc_reconnect)
        """
        raise NotImplementedError("WS/REST implementeres i næste PR")
