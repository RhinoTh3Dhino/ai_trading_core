# data/feed_orchestrator.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

from .gap_repair import rest_catchup
from .schemas import Bar
from .ws_utils import P99Tracker

log = logging.getLogger(__name__)

# Prøv at importere WS-moduler; hvis de mangler i CI, kør videre uden dem.
_binance_ws = _bybit_ws = None
try:
    from .exchanges import binance_ws as _binance_ws  # type: ignore
except Exception as e:
    log.debug("Kunne ikke importere data.exchanges.binance_ws: %r", e)

try:
    from .exchanges import bybit_ws as _bybit_ws  # type: ignore
except Exception as e:
    log.debug("Kunne ikke importere data.exchanges.bybit_ws: %r", e)

# Konfigurer rækkefølge/prioritet – du kan udvide med okx/kraken, når deres WS-moduler er klar
PRIMARY: List[str] = ["binance", "bybit"]
BACKUP: List[str] = ["okx", "kraken"]  # placeholder indtil WS-moduler findes

# Map over venues vi reelt understøtter lige nu (kun dem der kunne importeres)
SUBS: Dict[str, Callable] = {}
if _binance_ws is not None:
    SUBS["binance"] = _binance_ws.subscribe
if _bybit_ws is not None:
    SUBS["bybit"] = _bybit_ws.subscribe
# "okx" og "kraken" tilføjes når deres WS-moduler er klar


def _interval_ms(interval: str) -> int:
    """Konverter '1m','3m','1h','1d' -> millisekunder pr. bar."""
    s = interval.lower().strip()
    if s.endswith("ms"):
        return int(s[:-2])
    if s.endswith("s"):
        return int(s[:-1]) * 1_000
    if s.endswith("m"):
        return int(s[:-1]) * 60_000
    if s.endswith("h"):
        return int(s[:-1]) * 3_600_000
    if s.endswith("d"):
        return int(s[:-1]) * 86_400_000
    # fallback: antag minutter hvis kun tal
    if s.isdigit():
        return int(s) * 60_000
    raise ValueError(f"Ukendt interval-format: {interval}")


class FeedOrchestrator:
    """
    Multi-venue failover med hot-standby + kvalitetsscore.
    - Holder mindst 'min_active' venues kørende.
    - Reconnecter med eksponentiel backoff når en stream dør.
    - Måler p99-latens pr. venue og udfører REST catch-up ved bar-huller.
    - Stall-detektion: genstarter et venue, hvis der ikke er set data i lang tid.
    """

    def __init__(
        self,
        symbols: List[str],
        interval: str = "1m",
        min_active: int = 2,
        rest_lookback_bars: int = 120,
        max_backoff_sec: int = 30,
        stall_extra_ms: int = 30_000,  # ekstra margin oven på 3*bar_ms
    ):
        self.symbols = symbols
        self.interval = interval
        self.min_active = max(1, min_active)
        self.rest_lookback_bars = rest_lookback_bars
        self.max_backoff_sec = max_backoff_sec

        self.bar_ms: int = _interval_ms(interval)
        self.stall_threshold_ms: int = 3 * self.bar_ms + stall_extra_ms

        # Telemetri og status
        all_vs = list(PRIMARY) + list(BACKUP)
        self.p99: Dict[str, P99Tracker] = {v: P99Tracker() for v in all_vs}
        self.last_bar_ts: Dict[str, int] = (
            {}
        )  # key: f"{venue}:{symbol}" (bar close time)
        self.last_seen_wall_ms: Dict[str, int] = (
            {}
        )  # key: venue -> seneste modtagelsestid (wall clock)
        self.active: Dict[str, bool] = {v: False for v in all_vs}
        self.tasks: Dict[str, Optional[asyncio.Task]] = {v: None for v in all_vs}

    # ---------------------------- Internal helpers ----------------------------

    def _start_consumer(self, venue: str, out: asyncio.Queue):
        """Start en consumer-task for et venue, hvis den ikke allerede kører."""
        if venue not in SUBS:
            return
        t = self.tasks.get(venue)
        if t and not t.done():
            return  # kører allerede
        log.info("Starter feed-consumer for venue: %s", venue)
        task = asyncio.create_task(self._consume(venue, out))
        self.tasks[venue] = task

    async def _restart_consumer(self, venue: str, out: asyncio.Queue):
        """Afbryd og genstart en consumer-task (bruges ved stall eller hård fejl)."""
        if venue not in SUBS:
            return
        t = self.tasks.get(venue)
        if t and not t.done():
            log.warning("Genstarter venue %s (stall/fejl).", venue)
            t.cancel()
            try:
                await asyncio.wait_for(t, timeout=5.0)
            except Exception:
                pass
        self.active[venue] = False
        self._start_consumer(venue, out)

    def metrics_snapshot(self) -> Dict[str, float]:
        """Returnér p99-latens pr. venue som et snapshot (kan bruges i GUI/metrics)."""
        return {v: self.p99[v].p99() for v in self.p99}

    # ---------------------------- Consumer loop -------------------------------

    async def _consume(self, venue: str, queue: asyncio.Queue):
        """
        Robust konsum-loop for et venue:
        - kører subscribe-generatoren
        - måler latens
        - gap-repair via REST ved huller > 1 bar
        - auto-reconnect ved fejl
        """
        if venue not in SUBS:
            log.warning("Venue %s ikke understøttet endnu – ignoreres.", venue)
            self.active[venue] = False
            return

        subscribe = SUBS[venue]
        backoff = 1

        while True:
            try:
                async for bar in subscribe(self.symbols, self.interval):
                    key = f"{venue}:{bar.symbol}"
                    now_ms = int(time.time() * 1000)

                    # Latens-telemetri (bemærk: bar.ts = bar-close; ikke netværkslatens)
                    self.p99[venue].add(max(0, now_ms - int(bar.ts)))

                    # Stall-telemetri på venue-niveau (sidste wallclock-aktivitet)
                    self.last_seen_wall_ms[venue] = now_ms

                    # Gap-detektion
                    last = self.last_bar_ts.get(key)
                    if last is not None:
                        gap = int(bar.ts) - int(last)
                        if gap > self.bar_ms + 1:
                            missing_since = last + self.bar_ms
                            limit = min(
                                max(5, int(gap / self.bar_ms)), self.rest_lookback_bars
                            )
                            try:
                                for b in rest_catchup(
                                    bar.symbol,
                                    venue,
                                    self.interval,
                                    since_ms=missing_since,
                                    limit=limit,
                                ):
                                    await queue.put(b)
                            except Exception as e:
                                log.warning(
                                    "REST catch-up fejl for %s on %s: %s",
                                    bar.symbol,
                                    venue,
                                    repr(e),
                                )

                    # Opdater status og publicér bar
                    self.last_bar_ts[key] = int(bar.ts)
                    self.active[venue] = True
                    await queue.put(bar)

                # Hvis vi forlader async-for uden exception, betragtes stream som stoppet
                log.warning(
                    "Venue %s stream stoppede pænt – markerer inaktiv og forsøger reconnect.",
                    venue,
                )
                self.active[venue] = False

            except asyncio.CancelledError:
                # Graceful stop (fx Ctrl+C/ shutdown)
                log.info("Venue %s consumer annulleret.", venue)
                self.active[venue] = False
                break

            except Exception as e:
                log.warning("Venue %s stream fejlede: %s", venue, repr(e))
                self.active[venue] = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_sec)
                continue
            else:
                # Succes – reset backoff
                backoff = 1

    # ---------------------------- Public API ----------------------------------

    async def run(self) -> asyncio.Queue:
        """
        Returnerer en fælles Queue med Bars fra ≥ min_active venues (hot-standby).
        Starter PRIMARY, overvåger aktiv-status og aktiverer BACKUP hvis nødvendigt.
        """
        out: asyncio.Queue = asyncio.Queue(maxsize=5000)

        # Start alle PRIMARY venues vi understøtter (binance, bybit)
        for v in PRIMARY:
            self._start_consumer(v, out)

        async def _monitor():
            """Holder øje med aktive feeds, stall og forsøger at sikre min_active."""
            while True:
                now_ms = int(time.time() * 1000)  # kan bruges til fremtidig diagnostik
                active_count = sum(1 for v, a in self.active.items() if a)

                # Stall-detektion: hvis et venue ikke har set data længe, genstart
                for v in PRIMARY:
                    last_seen = self.last_seen_wall_ms.get(v)
                    if v in SUBS and (last_seen is not None):
                        if now_ms - last_seen > self.stall_threshold_ms:
                            await self._restart_consumer(v, out)

                # Sørg for at vi har mindst min_active venues kørende
                if active_count < self.min_active:
                    # Prøv først PRIMARY, derefter BACKUP i rækkefølge
                    for v in PRIMARY + BACKUP:
                        if v in SUBS and not self.active.get(v, False):
                            self._start_consumer(v, out)
                            # Vent kort og re-tæl
                            await asyncio.sleep(0.5)
                            active_count = sum(1 for _v, a in self.active.items() if a)
                            if active_count >= self.min_active:
                                break

                await asyncio.sleep(1.0)

        asyncio.create_task(_monitor())
        return out

    async def shutdown(self):
        """Afslut alle consumer-tasks pænt (bruges i services/live_connector.py i finally)."""
        tasks = [t for t in self.tasks.values() if t and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for v in self.active.keys():
            self.active[v] = False
