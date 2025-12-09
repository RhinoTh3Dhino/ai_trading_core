# bot/live_connector/runner.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


# --- Dev/emulering helpers ---------------------------------------------------
# Primær: brug ny, parametrisérbar emit_sample
try:
    from .metrics import emit_sample  # type: ignore
except Exception:

    def emit_sample(  # type: ignore
        venue: str,
        symbol: str = "TESTUSDT",
        transport_ms: Optional[float] = None,
        bar_close_lag_ms: Optional[float] = None,
        bars_inc: int = 1,
    ) -> dict:
        # Minimal fallback hvis metrics.emit_sample ikke findes
        now_ms = int(time.time() * 1000)
        observe_transport_latency(venue, symbol, now_ms - (transport_ms or 120))
        set_bar_close_lag(venue, symbol, now_ms - (bar_close_lag_ms or 1200))
        inc_bars(venue, symbol, max(1, bars_inc))
        try:
            with time_feature("EMA_14", symbol):
                pass
        except Exception:
            pass
        return {"ok": True, "venue": venue, "symbol": symbol}


# Valgfri router med GET/POST /_debug/emit_sample hvis tilgængelig i metrics
try:
    from .metrics import get_debug_router  # type: ignore
except Exception:

    def get_debug_router():  # type: ignore
        return None


# ----------------------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------------------
LOG = logging.getLogger("live_connector")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def env_bool(name: str, default: bool) -> bool:
    """Robust bool-læsning fra env (true/false, 1/0, yes/no, on/off)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    """Robust int-læsning fra env med fallback til default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("Ugyldig værdi for %s=%r – bruger default=%s", name, raw, default)
        return default


# QUIET / STATUS kommer nu fra LIVE_* så det matcher live.env + CLI/LiveConfig
QUIET = env_bool("LIVE_QUIET", default=True)
STATUS_MIN_SECS = env_int("LIVE_STATUS_MIN_SECS", default=30)

QUEUE_DEPTH_POLL_SECS = float(os.getenv("QUEUE_DEPTH_POLL_SECS", "2.0"))
READINESS_MAX_LAG_SECS = env_int("READINESS_MAX_LAG_SECS", default=120)

# Debug routes gate (default = ON for dev, som tidligere)
ENABLE_DEBUG_ROUTES = env_bool("ENABLE_DEBUG_ROUTES", default=True)
QUIET = os.getenv("QUIET", "1").strip().lower() not in {"0", "false", "no"}
STATUS_MIN_SECS = int(os.getenv("STATUS_MIN_SECS", "30"))
QUEUE_DEPTH_POLL_SECS = float(os.getenv("QUEUE_DEPTH_POLL_SECS", "2.0"))
READINESS_MAX_LAG_SECS = int(os.getenv("READINESS_MAX_LAG_SECS", "120"))

# Debug routes gate (default = ON for dev)
ENABLE_DEBUG_ROUTES = os.getenv("ENABLE_DEBUG_ROUTES", "1").strip().lower() not in {"0", "false"}

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
app = FastAPI(title="Live Connector", version="1.1.0")

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


_symbols_whitelist: List[str] = [
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

    # Monter valgfri debug-router (fra metrics), ellers fallback endpoints (nedenfor)
    if ENABLE_DEBUG_ROUTES:
        r = get_debug_router()
        if r is not None:
            app.include_router(r)

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
# Debug endpoints (fallback, kun hvis metrics-router ikke er monteret)
# ----------------------------------------------------------------------------------------
def _emit_sample_fallback_once(
    venue: str,
    symbol: str,
    transport_ms: Optional[float],
    bar_lag_ms: Optional[float],
) -> dict:
    now_ms = int(time.time() * 1000)
    # Brug primær emit_sample helper (eller dens fallback), så labels matches korrekt
    return emit_sample(
        venue=venue,
        symbol=symbol,
        transport_ms=transport_ms if transport_ms is not None else 120.0,
        bar_close_lag_ms=bar_lag_ms if bar_lag_ms is not None else 1200.0,
        bars_inc=1,
    )


def _debug_routes_enabled_but_no_router() -> bool:
    # True hvis vi vil have debugruter, men ingen router blev monteret i startup
    return ENABLE_DEBUG_ROUTES and get_debug_router() is None  # type: ignore


if _debug_routes_enabled_but_no_router():

    @app.get("/_debug/emit_sample")
    @app.post("/_debug/emit_sample")
    def _debug_emit_sample(
        venue: str = Query(default="bootstrap"),
        symbol: str = Query(default="TESTUSDT"),
        transport_ms: Optional[float] = Query(default=None),
        bar_lag_ms: Optional[float] = Query(default=None),
        n: int = Query(default=1, ge=0, le=100),
    ) -> JSONResponse:
        if not ENABLE_DEBUG_ROUTES:
            raise HTTPException(
                status_code=403, detail="Debug routes disabled (set ENABLE_DEBUG_ROUTES=1)"
            )
        out: dict = {}
        last: dict = {}
        for _ in range(max(1, n)):
            last = _emit_sample_fallback_once(venue, symbol, transport_ms, bar_lag_ms)
            out = last

        # Sørg for ready-state
        _active_venues[venue] = True
        _last_bar_ts_ms[symbol] = int(time.time() * 1000)
        return JSONResponse(out or {"ok": False, "error": "emit failed"})

    @app.post("/_debug/emit_n")
    def _debug_emit_n(
        n: int = Query(10, ge=1, le=1000, description="Antal samples at emulere sekventielt"),
        venue: str = Query(default="binance"),
        symbol: str = Query(default="TESTUSDT"),
    ) -> JSONResponse:
        if not ENABLE_DEBUG_ROUTES:
            raise HTTPException(
                status_code=403, detail="Debug routes disabled (set ENABLE_DEBUG_ROUTES=1)"
            )

        ok_count = 0
        for _ in range(n):
            r = _emit_sample_fallback_once(venue, symbol, None, None)
            ok_count += 1 if r.get("ok") else 0
            time.sleep(0.05)  # lille pacing til rate()-vinduer

        return JSONResponse({"ok": True, "emitted": ok_count})


# ----------------------------------------------------------------------------------------
# Lokal kørsel
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("bot.live_connector.runner:app", host="0.0.0.0", port=8000, workers=1)
