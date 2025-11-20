# bot/live_connector/venues/kraken.py
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional

from bot.live_connector.metrics import (
    inc_feed_bars_total,
    observe_bar_close_lag_ms,
    observe_transport_latency_ms,
)

from .base import BaseConnector

VENUE_NAME = "kraken"


def _to_internal_symbol(symbol_map: Dict[str, Dict[str, str]], kraken_pair: str) -> str:
    """
    Map Kraken pair (fx 'XBT/USDT' eller 'XBTUSDT') til internt symbol (fx 'BTCUSDT').
    """
    # Normaliser let: fjern slash for sammenligning
    k_norm = kraken_pair.replace("/", "")
    for internal, mapping in symbol_map.items():
        m = mapping.get("kraken") or ""
        if m.replace("/", "") == k_norm:
            return internal
    return k_norm  # defensiv fallback (f.eks. 'XBTUSDT')


def parse_kraken_candle_payload(
    msg: Dict[str, Any], symbol_map: Dict[str, Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """
    Normaliser et Kraken 1m OHLC payload til vores bar-event.
    Understøtter to former:

    A) 'ohlc-1'-kanal med array-række:
       {
         "channel": "ohlc-1",
         "pair": "XBT/USDT",
         "data": [[t, et, o, h, l, c, vwap, vol, count]]
       }

    B) 'ohlc' event med dict:
       {
         "event": "ohlc",
         "pair": "XBT/USDT",
         "interval": 1,
         "data": [{"time": 1730572740, "etime": 1730572800,
                   "open":"65000","high":"65100","low":"64900","close":"65050",
                   "vwap":"65040","vol":"10","count":42}]
       }
    """
    if not isinstance(msg, dict):
        return None

    # Find par & data
    pair = str(msg.get("pair") or msg.get("symbol") or msg.get("instId") or "") or None
    data = msg.get("data")

    if not pair or not data:
        return None
    if not isinstance(data, (list, tuple)) or not data:
        return None

    row = data[0]

    # Parse form A (array) eller form B (dict)
    try:
        if isinstance(row, (list, tuple)):
            # [t, et, o, h, l, c, vwap, vol, count]
            t_s, et_s = row[0], row[1]
            o, h, l, c = map(float, row[2:6])
            vol = float(row[7]) if len(row) > 7 else 0.0
        elif isinstance(row, dict):
            # {"time":t, "etime":et, "open":..., "high":..., "low":..., "close":..., "vol":...}
            t_s = row.get("time")
            et_s = row.get("etime") or (float(t_s) + 60.0)
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            vol = float(row.get("vol") or row.get("volume") or 0.0)
        else:
            return None
    except Exception:
        return None

    # timestamps kan være i sekunder (typisk Kraken) → konverter
    def _to_ms(ts_any: Any) -> int:
        tsf = float(ts_any)
        if tsf < 1e12:
            tsf *= 1000.0
        return int(tsf)

    open_ms = _to_ms(t_s)
    close_ms = _to_ms(et_s)
    symbol = _to_internal_symbol(symbol_map, pair)

    evt = {
        "venue": VENUE_NAME,
        "symbol": symbol,
        "tf": "1m",
        "open_time": open_ms,
        "close_time": close_ms,
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "v": vol,
        "is_final": True,  # vi publicerer kun finaliserede 1m-bars
        "event_ts_ms": int(time.time() * 1000),
    }
    return evt


class KrakenConnector(BaseConnector):
    venue = VENUE_NAME

    def __init__(
        self,
        cfg: dict,
        symbol_map: Dict[str, Dict[str, str]],
        on_kline: Optional[Callable[[dict], None]] = None,
        ws_client: Optional[Any] = None,
    ):
        if on_kline is None:
            on_kline = lambda _evt: None
        super().__init__(cfg=cfg, symbol_map=symbol_map, on_kline=on_kline, ws_client=ws_client)

    async def _subscribe(self, ws):
        """
        Forventet cfg["ws"]["subs"] ala:
          [{"channel":"ohlc-1","pair":"XBT/USDT"}, ...]
        """
        subs = [{"channel": s["channel"], "pair": s["pair"]} for s in self.cfg["ws"]["subs"]]
        await ws.send(json.dumps({"op": "subscribe", "args": subs}))

    async def _read_loop(self, ws):
        async for raw in ws:
            try:
                yield json.loads(raw)
            except Exception:
                continue

    def to_internal_symbol(self, kraken_pair: str) -> str:
        return _to_internal_symbol(self.symbol_map, kraken_pair)

    def _parse_kline(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        bar = parse_kraken_candle_payload(msg, self.symbol_map)
        if not bar:
            return None

        sym = bar["symbol"]
        now_ms = bar["event_ts_ms"]
        close_ms = bar["close_time"]

        inc_feed_bars_total(self.venue, sym)
        observe_bar_close_lag_ms(self.venue, sym, now_ms - close_ms)
        # Uden særskilt event_ts fra transportlaget bruger vi close_ms som proxy
        observe_transport_latency_ms(self.venue, sym, now_ms - close_ms)
        return bar
