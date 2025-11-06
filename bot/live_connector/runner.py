# bot/live_connector/runner.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response

# Prometheus endpoint/registry (ingen ny-registrering af metrics her!)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
    multiprocess,
)

# Projekt-metrics helpers (metrics er registreret i bot/live_connector/metrics.py)
from .metrics import (
    inc_bars,
    inc_reconnect,
    observe_transport_latency,
    set_bar_close_lag,
    set_queue_depth,
    time_feature,
)

# --- DQ helpers (robust import med no-op fallback) ---------------------------
try:
    from .metrics import inc_dq_violation, set_dq_freshness_minutes  # type: ignore
except Exception:  # pragma: no cover

    def inc_dq_violation(contract: str, rule: str) -> None:  # no-op fallback
        pass

    def set_dq_freshness_minutes(dataset: str, minutes: float) -> None:  # no-op
        pass


# --- Dev/emulering helper (til 5m rate/increase i dev) -----------------------
try:
    from .metrics import emit_sample_dev  # type: ignore
except Exception:

    def emit_sample_dev() -> dict:  # no-op hvis ikke tilgængelig
        return {"ok": False, "error": "emit_sample_dev not available"}


# ----------------------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------------------
LOG = logging.getLogger("live_connector")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

QUIET = os.getenv("QUIET", "1").strip().lower() not in {"0", "false", "no"}
STATUS_MIN_SECS = int(os.getenv("STATUS_MIN_SECS", "30"))
QUEUE_DEPTH_POLL_SECS = float(os.getenv("QUEUE_DEPTH_POLL_SECS", "2.0"))
READINESS_MAX_LAG_SECS = int(os.getenv("READINESS_MAX_LAG_SECS", "120"))

# Debug routes gate
ENABLE_DEBUG_ROUTES = os.getenv("ENABLE_DEBUG_ROUTES", "0").strip().lower() in {"1", "true"}

# Multiprocess-metrics?
PROMETHEUS_MULTIPROC_DIR = os.getenv("PROMETHEUS_MULTIPROC_DIR")

# DQ auth-secret til prod-endpoints
DQ_SHARED_SECRET = os.getenv("DQ_SHARED_SECRET", "").strip()


def _auth(secret: Optional[str]) -> bool:
    """Basic header-secret auth for /dq/* endpoints."""
    if not DQ_SHARED_SECRET:
        return True  # tillad lokalt/dev hvis ikke sat
    return (secret or "") == DQ_SHARED_SECRET


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
# Label guard (valgfri – robust fallback)
# ----------------------------------------------------------------------------------------
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


_symbols_whitelist = [
    s.strip() for s in os.getenv("OBS_SYMBOLS_WHITELIST", "").split(",") if s.strip()
]
_symbols_max = int(os.getenv("OBS_SYMBOLS_MAX", "100"))
SYMBOLS = LabelLimiter(whitelist=_symbols_whitelist or None, max_items=_symbols_max)


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
        return JSONResponse({"ready": False, "reason": "no-bars-seen-yet"}, status_code=503)
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
                _main_queue.qsize() if _main_queue and hasattr(_main_queue, "qsize") else None
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
# Robust relativ import; fallback til no-ops (tests skal ikke kræve feature-moduler)
try:  # pragma: no cover
    from features import (
        compute_all_features,
        compute_atr14,
        compute_ema14,
        compute_ema50,
        compute_rsi14,
        compute_vwap,
    )
except Exception:  # pragma: no cover
    compute_all_features = None
    compute_ema14 = compute_ema50 = compute_rsi14 = compute_vwap = compute_atr14 = None


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
    interval = max(5, STATUS_MIN_SECS)
    while True:
        try:
            if not QUIET:
                now = time.time()
                if (now - _last_status_log_ms) >= interval:
                    _last_status_log_ms = now
                    newest = max(_last_bar_ts_ms.values()) if _last_bar_ts_ms else 0
                    lag_ms = int(time.time() * 1000) - newest if newest else None
                    LOG.info(
                        "STATUS venues=%s symbols=%d lag_ms=%s queue=%s",
                        ",".join(sorted(k for k, v in _active_venues.items() if v)) or "-",
                        len(_last_bar_ts_ms),
                        lag_ms if lag_ms is not None else "-",
                        (
                            _main_queue.qsize()
                            if _main_queue and hasattr(_main_queue, "qsize")
                            else "-"
                        ),
                    )
            await asyncio.sleep(interval)
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
        "Live Connector startup (QUIET=%s, STATUS_MIN_SECS=%s, ENABLE_DEBUG_ROUTES=%s)",
        QUIET,
        STATUS_MIN_SECS,
        ENABLE_DEBUG_ROUTES,
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
# PROD DQ endpoints (med header-secret)
# ----------------------------------------------------------------------------------------
@app.post("/dq/freshness")
def dq_freshness_update(
    dataset: str = Query(...),
    minutes: float = Query(...),
    x_dq_secret: Optional[str] = Header(default=None),
) -> JSONResponse:
    if not _auth(x_dq_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        set_dq_freshness_minutes(dataset, float(minutes))
        return JSONResponse({"ok": True, "dataset": dataset, "minutes": float(minutes)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dq/violation")
def dq_violation_inc(
    contract: str = Query(...),
    rule: str = Query(...),
    n: int = Query(default=1, ge=1),
    x_dq_secret: Optional[str] = Header(default=None),
) -> JSONResponse:
    if not _auth(x_dq_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    for _ in range(n):
        inc_dq_violation(contract, rule)
    return JSONResponse({"ok": True, "contract": contract, "rule": rule, "inc": n})


# ----------------------------------------------------------------------------------------
# Debug endpoints (bag feature-flag)
# ----------------------------------------------------------------------------------------
def _emit_sample_fallback() -> dict:
    """
    Lokal fallback, der emulerer én bar + transport-latency + en feature-timing.
    Giver liv i:
      - feed_bars_total
      - feed_transport_latency_ms_bucket
      - feature_compute_ms_bucket
      - feed_bar_close_lag_ms
    """
    now_ms = int(time.time() * 1000)
    venue = "binance"
    symbol = "BTCUSDT"

    observe_transport_latency(venue, symbol, now_ms - 120)
    set_bar_close_lag(venue, symbol, now_ms - 1500)
    inc_bars(venue, symbol, 1)

    # Minimal feature-timing (måler bare context-manager overhead)
    try:
        with time_feature("EMA_14", symbol):
            pass
    except Exception:
        pass

    _active_venues[venue] = True
    _last_bar_ts_ms[symbol] = now_ms
    set_queue_depth(5, "live")

    return {"ok": True, "source": "fallback", "venue": venue, "symbol": "BTCUSDT"}


@app.post("/_debug/emit_sample")
def _debug_emit_sample() -> JSONResponse:
    if not ENABLE_DEBUG_ROUTES:
        raise HTTPException(
            status_code=403, detail="Debug routes disabled (set ENABLE_DEBUG_ROUTES=1)"
        )

    # Brug central dev-helper hvis muligt
    result = emit_sample_dev()
    if not result.get("ok"):
        result = _emit_sample_fallback()

    # Sørg for ready-state (i tilfælde af helt tom proces)
    if "BTCUSDT" not in _last_bar_ts_ms:
        _last_bar_ts_ms["BTCUSDT"] = int(time.time() * 1000)
    _active_venues.setdefault("binance", True)

    return JSONResponse(result)


@app.post("/_debug/emit_n")
def _debug_emit_n(
    n: int = Query(10, ge=1, le=1000, description="Antal samples at emulere sekventielt")
) -> JSONResponse:
    if not ENABLE_DEBUG_ROUTES:
        raise HTTPException(
            status_code=403, detail="Debug routes disabled (set ENABLE_DEBUG_ROUTES=1)"
        )

    ok_count = 0
    for _ in range(n):
        r = emit_sample_dev()
        if not r.get("ok"):
            r = _emit_sample_fallback()
        ok_count += 1 if r.get("ok") else 0
        # let pacing så Prometheus rate() har vindue at arbejde med
        time.sleep(0.05)

    return JSONResponse({"ok": True, "emitted": ok_count})


@app.post("/debug/dq")
def debug_dq(
    contract: str = Query(default="ohlcv_1h"),
    rule: str = Query(default="bounds_min"),
    freshness: Optional[float] = Query(default=None, description="Minutter siden seneste update"),
    inc: int = Query(default=0, description="Inkrementér violations N gange"),
) -> JSONResponse:
    """
    Sæt DQ-metrikker direkte i denne app-proces (til test/observability).
    Beskyttet af ENABLE_DEBUG_ROUTES=1.
    """
    if not ENABLE_DEBUG_ROUTES:
        raise HTTPException(
            status_code=403, detail="Debug routes disabled (set ENABLE_DEBUG_ROUTES=1)"
        )

    changed: Dict[str, Any] = {}
    if freshness is not None:
        set_dq_freshness_minutes("ohlcv_1h", float(freshness))
        changed["dq_freshness_minutes"] = float(freshness)
    if inc and inc > 0:
        for _ in range(int(inc)):
            inc_dq_violation(contract, rule)
        changed["dq_violations_total"] = {"contract": contract, "rule": rule, "inc": int(inc)}

    return JSONResponse({"ok": True, "changed": changed})


# ----------------------------------------------------------------------------------------
# Lokal kørsel
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("bot.live_connector.runner:app", host="0.0.0.0", port=8000, workers=1)
