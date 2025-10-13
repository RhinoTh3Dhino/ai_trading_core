# bot/live_connector/venues/okx.py
import json
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from bot.live_connector.metrics import (
    inc_feed_bars_total,
    observe_bar_close_lag_ms,
    observe_transport_latency_ms,
)

from .base import BaseConnector

VENUE_NAME = "okx"

SymbolMap = Mapping[str, Mapping[str, str]]
SymbolResolver = Callable[[str], str]
SymbolMapOrResolver = Union[SymbolMap, SymbolResolver]


def _resolve_internal_symbol(symbol_map_or_resolver: SymbolMapOrResolver, inst_id: str) -> str:
    """
    Returnér internt symbol for et OKX instId.
    Understøtter både dict-symbolmap og en resolver-funktion (callable).
    """
    if callable(symbol_map_or_resolver):
        try:
            sym = symbol_map_or_resolver(inst_id)
            return str(sym) if sym else inst_id.replace("-", "")
        except Exception:
            return inst_id.replace("-", "")

    try:
        for internal, mapping in symbol_map_or_resolver.items():
            if isinstance(mapping, Mapping) and mapping.get("okx") == inst_id:
                return internal
    except Exception:
        pass

    return inst_id.replace("-", "")  # defensiv fallback


def parse_okx_candle_payload(
    msg: Dict[str, Any], symbol_map_or_resolver: SymbolMapOrResolver
) -> Optional[List[Dict[str, Any]]]:
    """
    Parser OKX candle1m payload → liste af normaliserede kline-events.
    Forventer fx:
      {"arg":{"channel":"candle1m","instId":"BTC-USDT"},
       "data":[["1730572800000","65000","65100","64900","65050","10","0","0","1"], ...]}

    Returnerer:
      [ {venue,symbol,tf,open_time,close_time,o,h,l,c,v,is_final,event_ts_ms}, ... ]
    eller None hvis payload ikke er relevant/valid.
    """
    if not isinstance(msg, dict) or "data" not in msg or "arg" not in msg:
        return None

    arg = msg["arg"]
    if not isinstance(arg, dict) or arg.get("channel") != "candle1m":
        return None

    data = msg["data"]
    if not (isinstance(data, list) and data and isinstance(data[0], (list, tuple))):
        return None

    inst_id = str(arg.get("instId") or "")
    symbol = _resolve_internal_symbol(symbol_map_or_resolver, inst_id)
    now_ms = int(time.time() * 1000)

    events: List[Dict[str, Any]] = []
    for row in data:
        try:
            # OKX: [close_ts_ms, o, h, l, c, vol, volC? ...]
            ts_ms = int(float(row[0]))  # close time i ms
            o, h, l, c = map(float, row[1:5])
            v = float(row[6] if len(row) > 6 else row[5])
        except Exception:
            continue

        events.append(
            {
                "venue": VENUE_NAME,
                "symbol": symbol,
                "tf": "1m",
                "open_time": ts_ms - 60_000,
                "close_time": ts_ms,
                "o": o,
                "h": h,
                "l": l,
                "c": c,
                "v": v,
                "is_final": True,  # candle1m leverer normalt afsluttede lys
                "event_ts_ms": now_ms,
            }
        )

    return events if events else None


def parse_okx_candle_payload_one(
    msg: Dict[str, Any], symbol_map_or_resolver: SymbolMapOrResolver
) -> Optional[Dict[str, Any]]:
    """
    Convenience: returnér det første parse-de event (eller None).
    Bruges af connectoren, som forventer ét event ad gangen.
    """
    events = parse_okx_candle_payload(msg, symbol_map_or_resolver)
    if not events:
        return None
    return events[0]


class OKXConnector(BaseConnector):
    venue = VENUE_NAME

    # Gør on_kline/ws_client valgfrie for test-kompatibilitet
    def __init__(
        self,
        cfg: dict,
        symbol_map: SymbolMapOrResolver,
        on_kline: Optional[Callable[[dict], None]] = None,
        ws_client: Optional[Any] = None,
    ):
        if on_kline is None:
            on_kline = lambda _evt: None  # no-op i tests
        super().__init__(cfg=cfg, symbol_map=symbol_map, on_kline=on_kline, ws_client=ws_client)

    async def _subscribe(self, ws):
        subs = [{"channel": s["channel"], "instId": s["instId"]} for s in self.cfg["ws"]["subs"]]
        await ws.send(json.dumps({"op": "subscribe", "args": subs}))

    async def _read_loop(self, ws):
        async for raw in ws:
            try:
                yield json.loads(raw)
            except Exception:
                continue

    # Public metode forventet af tests
    def to_internal_symbol(self, inst_id: str) -> str:
        return _resolve_internal_symbol(self.symbol_map, inst_id)

    # Thin wrapper (bevares for bagudkompatibilitet)
    def _to_internal(self, instId: str) -> str:
        return _resolve_internal_symbol(self.symbol_map, instId)

    def _parse_kline(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Deleger til parser-one og instrumentér metrics for det ene event.
        """
        bar = parse_okx_candle_payload_one(msg, self.symbol_map)
        if not bar:
            return None

        # Metrics (per venue/symbol)
        sym = bar["symbol"]
        now_ms = bar["event_ts_ms"]
        close_ms = bar["close_time"]

        inc_feed_bars_total(self.venue, sym)
        observe_bar_close_lag_ms(self.venue, sym, now_ms - close_ms)
        # Vi har ikke separat event_ts fra netlaget; brug close_time som proxy
        observe_transport_latency_ms(self.venue, sym, now_ms - close_ms)

        return bar
