# bot/live_connector/runner.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
# Prometheus endpoint/registry (ingen ny-registrering af metrics her!)
from prometheus_client import (CONTENT_TYPE_LATEST, REGISTRY,
                               CollectorRegistry, generate_latest,
                               multiprocess)

# Projekt-metrics helpers (metrics er registreret i bot/live_connector/metrics.py)
from .metrics import (inc_bars, inc_reconnect, observe_transport_latency,
                      set_bar_close_lag, set_queue_depth, time_feature)

# Label guard (valgfri)
try:
    from .label_guard import LabelLimiter  # hvis tilgængelig i repo
except Exception:  # pragma: no cover

    class LabelLimiter:  # no-op fallback
        def __init__(self, whitelist=None, max_items: int = 10_000):
            self.whitelist = set(whitelist or [])
            self.max_items = max_items
            self.seen = set()

        def allow(self, value: str) -> bool:
            if self.whitelist and value not in self.whitelist:
                return False
            if value in self.seen:
                return True
            if len(self.seen) >= self.max_items:
                return False
            self.seen.add(value)
            return True


# Feature-API (best effort)
try:  # pragma: no cover
    from .features import (compute_all_features, compute_atr14, compute_ema14,
                           compute_ema50, compute_rsi14, compute_vwap)
except Exception:  # pragma: no cover
    compute_all_features = None
    compute_ema14 = compute_ema50 = compute_rsi14 = compute_vwap = compute_atr14 = None


# ----------------------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------------------
LOG = logging.getLogger("live_connector")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

QUIET = os.getenv("QUIET", "1") not in ("0", "false", "False", "no", "NO")
STATUS_MIN_SECS = int(os.getenv("STATUS_MIN_SECS", "30"))
QUEUE_DEPTH_POLL_SECS = float(os.getenv("QUEUE_DEPTH_POLL_SECS", "2.0"))
READINESS_MAX_LAG_SECS = int(os.getenv("READINESS_MAX_LAG_SECS", "120"))

# Label-guard setup
_symbols_whitelist = [
    s.strip() for s in os.getenv("OBS_SYMBOLS_WHITELIST", "").split(",") if s.strip()
]
_symbols_max = int(os.getenv("OBS_SYMBOLS_MAX", "100"))
SYMBOLS = LabelLimiter(whitelist=_symbols_whitelist or None, max_items=_symbols_max)

# Multiprocess-metrics?
PROMETHEUS_MULTIPROC_DIR = os.getenv("PROMETHEUS_MULTIPROC_DIR")


# ----------------------------------------------------------------------------------------
# Datamodeller
# ----------------------------------------------------------------------------------------
@dataclass
class Msg:
    venue: str
    symbol: str
    event_ts_ms: Optional[int]


@dataclass
class Bar:
    venue: str
    symbol: str
    end_ms: int
    is_final: bool = True
    payload: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------------------------
# App & /metrics endpoint
# ----------------------------------------------------------------------------------------
app = FastAPI(title="Live Connector", version="1.0.0")

if PROMETHEUS_MULTIPROC_DIR:
    _registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(_registry)

    @app.get("/metrics")
    async def metrics_mp() -> Response:
        data = generate_latest(_registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

else:

    @app.get("/metrics")
    async def metrics_sp() -> Response:
        data = generate_latest(REGISTRY)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ----------------------------------------------------------------------------------------
# Interne state
# ----------------------------------------------------------------------------------------
_last_bar_ts_ms: Dict[str, int] = {}
_last_status_log_ms: float = 0.0
_active_venues: Dict[str, bool] = {}
_main_queue: Optional[Any] = None


# ----------------------------------------------------------------------------------------
# Health/Ready/Status
# ----------------------------------------------------------------------------------------
@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": "live_connector", "status": "ok"})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/ready")
async def ready() -> JSONResponse:
    if not _last_bar_ts_ms:
        return JSONResponse(
            {"ready": False, "reason": "no-bars-seen-yet"}, status_code=503
        )
    now_ms = int(time.time() * 1000)
    newest = max(_last_bar_ts_ms.values())
    lag_ms = now_ms - newest
    ok = lag_ms <= READINESS_MAX_LAG_SECS * 1000
    return JSONResponse({"ready": ok, "lag_ms": lag_ms}, status_code=200 if ok else 503)


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse(
        {
            "venues": _active_venues,
            "symbols": len(_last_bar_ts_ms),
            "last_bar_ts_ms": _last_bar_ts_ms,
            "queue_depth": (
                _main_queue.qsize()
                if _main_queue and hasattr(_main_queue, "qsize")
                else None
            ),
            "quiet": QUIET,
        }
    )


# ----------------------------------------------------------------------------------------
# Hooks fra streaming/orchestrator
# ----------------------------------------------------------------------------------------
async def on_tick_or_kline(msg: Msg) -> None:
    _active_venues[msg.venue] = True
    if SYMBOLS.allow(msg.symbol):
        observe_transport_latency(msg.venue, msg.symbol, msg.event_ts_ms)


async def on_bar_final(bar: Bar) -> None:
    if not bar.is_final:
        return
    _last_bar_ts_ms[bar.symbol] = int(bar.end_ms)
    if SYMBOLS.allow(bar.symbol):
        set_bar_close_lag(bar.venue, bar.symbol, bar.end_ms)
        inc_bars(bar.venue, bar.symbol, 1)


def on_reconnect(venue: str) -> None:
    _active_venues[venue] = True
    inc_reconnect(venue)


def register_main_queue(q: Any) -> None:
    global _main_queue
    _main_queue = q


async def poll_queue_depth(q: Any) -> None:
    depth = q.qsize() if hasattr(q, "qsize") else None
    if depth is not None:
        set_queue_depth(int(depth), queue_name="live")


# ----------------------------------------------------------------------------------------
# Feature-beregning
# ----------------------------------------------------------------------------------------
async def compute_features_for_bar(bar: Bar) -> None:
    symbol = bar.symbol
    ran_any = False
    if callable(compute_ema14):
        ran_any = True
        with time_feature("EMA_14", symbol):
            compute_ema14(bar)
    if callable(compute_ema50):
        ran_any = True
        with time_feature("EMA_50", symbol):
            compute_ema50(bar)
    if callable(compute_rsi14):
        ran_any = True
        with time_feature("RSI_14", symbol):
            compute_rsi14(bar)
    if callable(compute_vwap):
        ran_any = True
        with time_feature("VWAP", symbol):
            compute_vwap(bar)
    if callable(compute_atr14):
        ran_any = True
        with time_feature("ATR_14", symbol):
            compute_atr14(bar)

    if not ran_any and callable(compute_all_features):
        with time_feature("ALL", symbol):
            compute_all_features(bar)


# ----------------------------------------------------------------------------------------
# Baggrundstasks
# ----------------------------------------------------------------------------------------
async def _bg_status_task() -> None:
    global _last_status_log_ms
    while True:
        try:
            if not QUIET:
                now = time.time()
                if (now - _last_status_log_ms) >= max(5, STATUS_MIN_SECS):
                    _last_status_log_ms = now
                    newest = max(_last_bar_ts_ms.values()) if _last_bar_ts_ms else 0
                    lag_ms = int(time.time() * 1000) - newest if newest else None
                    LOG.info(
                        "STATUS venues=%s symbols=%d lag_ms=%s queue=%s",
                        ",".join(sorted(k for k, v in _active_venues.items() if v))
                        or "-",
                        len(_last_bar_ts_ms),
                        lag_ms if lag_ms is not None else "-",
                        (
                            _main_queue.qsize()
                            if _main_queue and hasattr(_main_queue, "qsize")
                            else "-"
                        ),
                    )
            await asyncio.sleep(max(5, STATUS_MIN_SECS))
        except asyncio.CancelledError:
            return
        except Exception as e:  # pragma: no cover
            LOG.warning("status task error: %s", e)
            await asyncio.sleep(5)


async def _bg_queue_depth_task() -> None:
    while True:
        try:
            if _main_queue and hasattr(_main_queue, "qsize"):
                set_queue_depth(int(_main_queue.qsize()), queue_name="live")
            await asyncio.sleep(QUEUE_DEPTH_POLL_SECS)
        except asyncio.CancelledError:
            return
        except Exception as e:  # pragma: no cover
            LOG.warning("queue depth task error: %s", e)
            await asyncio.sleep(1.0)


# ----------------------------------------------------------------------------------------
# Lifespan hooks
# ----------------------------------------------------------------------------------------
_bg_tasks: list[asyncio.Task] = []


@app.on_event("startup")
async def _on_startup() -> None:
    LOG.info(
        "Live Connector startup (QUIET=%s, STATUS_MIN_SECS=%s)", QUIET, STATUS_MIN_SECS
    )

    # Prime nogle metrics via helpers (ingen ny-registrering)
    set_queue_depth(0, "live")
    now_ms = int(time.time() * 1000)
    venue = "bootstrap"
    symbol = "TESTUSDT"

    # Sørger for at 'feed_transport_latency_ms_bucket' m.m. materialiseres
    observe_transport_latency(venue, symbol, now_ms - 5)
    set_bar_close_lag(venue, symbol, now_ms - 1500)
    inc_bars(venue, symbol, 1)
    try:
        with time_feature("EMA_14", symbol):
            pass
    except Exception:
        pass

    _active_venues[venue] = True
    _last_bar_ts_ms[symbol] = now_ms

    _bg_tasks.append(asyncio.create_task(_bg_status_task(), name="status"))
    _bg_tasks.append(asyncio.create_task(_bg_queue_depth_task(), name="queue-depth"))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    LOG.info("Live Connector shutdown")
    for t in _bg_tasks:
        t.cancel()
    for t in _bg_tasks:
        try:
            await t
        except BaseException:
            pass
    _bg_tasks.clear()


# ----------------------------------------------------------------------------------------
# Debug endpoints
# ----------------------------------------------------------------------------------------
@app.post("/_debug/emit_sample")
def _debug_emit_sample() -> JSONResponse:
    now_ms = int(time.time() * 1000)
    venue = "binance"
    symbol = "BTCUSDT"

    observe_transport_latency(venue, symbol, now_ms - 120)
    set_bar_close_lag(venue, symbol, now_ms - 1500)
    inc_bars(venue, symbol, 1)
    set_queue_depth(5, "live")

    _active_venues[venue] = True
    _last_bar_ts_ms[symbol] = now_ms

    return JSONResponse({"ok": True})


# ----------------------------------------------------------------------------------------
# Lokal kørsel
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("bot.live_connector.runner:app", host="0.0.0.0", port=8000, workers=1)
