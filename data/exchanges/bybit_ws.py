# data/exchanges/bybit_ws.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, List, Optional, Dict

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from data.schemas import Bar

log = logging.getLogger(__name__)

# Dæmp 3.-parts støj (fjern disse linjer hvis du vil styre det globalt i din app)
for _name in ("websockets", "websockets.client", "websockets.protocol"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Bybit v5 WS endpoints (public)
BYBIT_WS_LINEAR = "wss://stream.bybit.com/v5/public/linear"  # USDT perpetuals
BYBIT_WS_SPOT   = "wss://stream.bybit.com/v5/public/spot"    # Spot

PING_INTERVAL_SEC = 20
PING_TIMEOUT_SEC  = 10
PROBE_TIMEOUT_SEC = 12     # hvis ingen kline-data inden dette → prøv anden kategori
MAX_BACKOFF_SEC   = 30
CHUNK_SIZE        = 10     # Bybit begrænser args til ≤10 pr. subscribe


def _bybit_interval(interval: str) -> str:
    """
    Map '1m' -> '1', '3m' -> '3', '5m' -> '5', '1h' -> '60', '1d' -> '1440'.
    (Bybit accepterer også 'D', men vi holder os til minutter for konsistens.)
    """
    s = interval.lower().strip()
    if s.endswith("m"):  # minutter
        return s[:-1]
    if s.endswith("h"):  # timer -> minutter
        return str(int(s[:-1]) * 60)
    if s.endswith("d"):  # dage -> minutter
        return str(int(s[:-1]) * 60 * 24)
    return s


def _topics(symbols: List[str], interval: str) -> List[str]:
    iv = _bybit_interval(interval)
    return [f"kline.{iv}.{sym.upper()}" for sym in symbols]


def _chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


async def _stream_from_url(
    url: str,
    symbols: List[str],
    interval: str,
    include_partials: bool,
) -> AsyncIterator[Bar]:
    """
    Åbn WS mod 'url', subscribe til alle kline-topics (chunket ≤10),
    vent på første kline (med PROBE-timeout), og stream derefter kontinuerligt.

    include_partials=False → emit kun lukkede lys (confirm=True).
    """
    topics = _topics(symbols, interval)

    # Dedup på ts pr. symbol (undgå dobbelt close-events ved reconnect/bursts)
    last_ts: Dict[str, int] = {}

    async with websockets.connect(
        url,
        ping_interval=PING_INTERVAL_SEC,   # TCP-level ping frames
        ping_timeout=PING_TIMEOUT_SEC,
        max_size=2_000_000,
        close_timeout=1,
    ) as ws:
        # Subscribe i chunks
        for chunk in _chunked(topics, CHUNK_SIZE):
            await ws.send(json.dumps({"op": "subscribe", "args": chunk}))

        first_kline_received = False
        probe_deadline = asyncio.get_event_loop().time() + PROBE_TIMEOUT_SEC

        while True:
            # Under probe: tidsbegrænset recv
            if not first_kline_received:
                timeout = max(0.0, probe_deadline - asyncio.get_event_loop().time())
                if timeout == 0.0:
                    raise asyncio.TimeoutError("Bybit: ingen kline-data i probe-vindue")
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            else:
                raw = await ws.recv()

            # Parse JSON
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            # Ignorér kontrol-/ack-beskeder
            if isinstance(msg, dict) and msg.get("op") in {"subscribe", "ping", "pong"}:
                continue

            topic = msg.get("topic", "")
            if not topic.startswith("kline."):
                continue

            # topic-format: kline.{iv}.{SYMBOL}
            try:
                symbol_from_topic = topic.split(".")[-1].upper()
            except Exception:
                log.debug("Bybit: kunne ikke parse symbol fra topic=%s", topic)
                continue

            data = msg.get("data") or []
            if data and not first_kline_received:
                first_kline_received = True
                log.info("Bybit [%s] første kline-data modtaget (topic=%s)", url, topic)

            for d in data:
                try:
                    # V5-felter: start, end (ms), open, high, low, close, volume, turnover, confirm
                    ts_end = int(d.get("end") or d.get("start"))
                    is_final = bool(d.get("confirm", False))  # True når bar er lukket

                    if (not include_partials) and (not is_final):
                        continue  # drop in-progress for roligt output

                    # Robust symbol: brug feltet hvis til stede, ellers parse fra topic
                    symbol = (d.get("symbol") or symbol_from_topic).upper()

                    # Dedupér lukkede lys på (symbol, ts)
                    if is_final and last_ts.get(symbol) == ts_end:
                        continue
                    if is_final:
                        last_ts[symbol] = ts_end

                    yield Bar(
                        venue="bybit",
                        symbol=symbol,
                        ts=ts_end,
                        interval=interval,                     # kræves af din Bar-model
                        open=float(d["open"]),
                        high=float(d["high"]),
                        low=float(d["low"]),
                        close=float(d["close"]),
                        volume=float(d["volume"]),
                        is_final=is_final,
                    )
                except Exception as e:
                    # Kun DEBUG for at undgå støj ved enkelte records
                    log.debug(
                        "Bybit parse-fejl for %s: %r (payload=%s)",
                        symbol_from_topic, e, d
                    )


async def subscribe(
    symbols: List[str],
    interval: str = "1m",
    include_partials: bool = False,
) -> AsyncIterator[Bar]:
    """
    Auto-probe: prøv LINEAR først; hvis ingen kline i PROBE-timeout, fald tilbage til SPOT.
    Ved fejl reconnectes med eksponentiel backoff.

    include_partials=False (default) → kun lukkede lys for roligere output.
    """
    preferred_url = BYBIT_WS_LINEAR
    fallback_url  = BYBIT_WS_SPOT

    chosen = preferred_url

    # Probe: prøv valgt kategori én gang
    try:
        async for bar in _stream_from_url(chosen, symbols, interval, include_partials):
            yield bar  # LINEAR virker → fortsæt på LINEAR
    except asyncio.TimeoutError:
        log.warning("Bybit LINEAR gav ingen data på ~%ss → forsøger SPOT", PROBE_TIMEOUT_SEC)
        chosen = fallback_url
    except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError, OSError) as e:
        log.warning("Bybit (%s) lukket under probe: %r → prøver samme igen", chosen, e)
    except Exception as e:
        log.warning("Bybit ukendt fejl under probe (%s): %r → prøver samme igen", chosen, e)

    # Hovedloop (reconnect + backoff)
    backoff = 1
    while True:
        try:
            async for bar in _stream_from_url(chosen, symbols, interval, include_partials):
                yield bar
            # Hvis stream stopper uden exception, tving en reconnect
            raise ConnectionClosedError(None, None, None)
        except asyncio.TimeoutError:
            # Flip kategori ved gentagen timeout
            chosen = fallback_url if chosen == preferred_url else preferred_url
            log.info("Bybit skifter kategori efter timeout → %s", "SPOT" if chosen == fallback_url else "LINEAR")
        except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError, OSError) as e:
            log.warning("Bybit WS (%s) lukket/fejl: %r", chosen, e)
        except Exception as e:
            log.warning("Bybit ukendt fejl (%s): %r", chosen, e)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF_SEC)
