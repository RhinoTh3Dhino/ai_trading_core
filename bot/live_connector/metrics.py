# bot/live_connector/metrics.py
"""
Prometheus metrics til live-connectoren.

- Navne matcher recording rules / dashboards.
- Idempotent og reload-sikker registrering via ensure_registered().
- make_metrics_app() kalder ensure_registered() og bootstrapper som standard,
  så /metrics eksponerer histogram/gauge/counter-serier – også uden live-feed.
- Helper-funktioner kalder defensivt ensure_registered() for at undgå race.

Fase 5 (DQ) + Fase 6 (Soak/Chaos) klar.

NYT:
- emit_sample(venue, symbol, ...) som emulerer trafik for et konkret venue.
- Valgfrit FastAPI-router (get_debug_router) med /_debug/emit_sample (GET/POST),
  aktiveres hvis ENABLE_DEBUG_ROUTES != 0/false.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, make_asgi_app

# Multiprocess flag
_MULTIPROC = bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))

# Bootstrap tomme serier ved start?
_BOOTSTRAP = os.getenv("METRICS_BOOTSTRAP", "1").strip().lower() not in {"0", "false"}

# ms-buckets (dækker både lav latenstid og spikes)
_MS_BUCKETS = (
    1,
    2,
    5,
    10,
    25,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1_000,
    1_500,
    2_000,
    3_000,
    5_000,
    7_500,
    10_000,
)

# Globals (sættes ved ensure_registered)
feed_transport_latency_ms: Histogram | None = None
feed_bar_close_lag_ms: Gauge | None = None
feed_bars_total: Counter | None = None
feed_reconnects_total: Counter | None = None
feed_queue_depth: Gauge | None = None
feature_compute_ms: Histogram | None = None
feature_errors_total: Counter | None = None

# Fase 5 — DQ
dq_violations_total: Counter | None = None
dq_freshness_minutes: Gauge | None = None

_METRICS_READY = False
_BOOTSTRAPPED = False


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(ts: Union[int, float]) -> int:
    tsf = float(ts)
    if tsf < 1e12:
        tsf *= 1000.0
    return int(tsf)


def _gauge_kwargs() -> dict:
    return {"multiprocess_mode": "max"} if _MULTIPROC else {}


def _registry_lookup(name: str) -> Any | None:
    try:
        mapping: Dict[str, Any] = getattr(REGISTRY, "_names_to_collectors", {})  # type: ignore[attr-defined]
        return mapping.get(name)
    except Exception:
        return None


def ensure_registered() -> None:
    """Opret (én gang) alle metrikker. Reload-sikker."""
    global _METRICS_READY
    if _METRICS_READY:
        return

    global feed_transport_latency_ms, feed_bar_close_lag_ms, feed_bars_total
    global feed_reconnects_total, feed_queue_depth, feature_compute_ms, feature_errors_total
    global dq_violations_total, dq_freshness_minutes

    # Rebind hvis de allerede findes
    existing = {
        "feed_transport_latency_ms": _registry_lookup("feed_transport_latency_ms"),
        "feed_bar_close_lag_ms": _registry_lookup("feed_bar_close_lag_ms"),
        "feed_bars_total": _registry_lookup("feed_bars_total"),
        "feed_reconnects_total": _registry_lookup("feed_reconnects_total"),
        "feed_queue_depth": _registry_lookup("feed_queue_depth"),
        "feature_compute_ms": _registry_lookup("feature_compute_ms"),
        "feature_errors_total": _registry_lookup("feature_errors_total"),
        "dq_violations_total": _registry_lookup("dq_violations_total"),
        "dq_freshness_minutes": _registry_lookup("dq_freshness_minutes"),
    }
    if all(v is not None for v in existing.values()):
        feed_transport_latency_ms = existing["feed_transport_latency_ms"]
        feed_bar_close_lag_ms = existing["feed_bar_close_lag_ms"]
        feed_bars_total = existing["feed_bars_total"]
        feed_reconnects_total = existing["feed_reconnects_total"]
        feed_queue_depth = existing["feed_queue_depth"]
        feature_compute_ms = existing["feature_compute_ms"]
        feature_errors_total = existing["feature_errors_total"]
        dq_violations_total = existing["dq_violations_total"]
        dq_freshness_minutes = existing["dq_freshness_minutes"]
        _METRICS_READY = True
        return

    # Førstegangs-registrering
    feed_transport_latency_ms = Histogram(
        "feed_transport_latency_ms",
        "End-to-end transportlatens (ms): now_ms - event_ts fra venue besked",
        labelnames=("venue", "symbol"),
        buckets=_MS_BUCKETS,
    )
    feed_bar_close_lag_ms = Gauge(
        "feed_bar_close_lag_ms",
        "Hvor langt inde i baren vi er: now_ms - bar_end_ms",
        labelnames=("venue", "symbol"),
        **_gauge_kwargs(),
    )
    feed_bars_total = Counter(
        "feed_bars_total",
        "Antal lukkede bars (is_final=True) modtaget/produceret",
        labelnames=("venue", "symbol"),
    )
    feed_reconnects_total = Counter(
        "feed_reconnects_total",
        "Antal WS reconnects per venue",
        labelnames=("venue",),
    )
    feed_queue_depth = Gauge(
        "feed_queue_depth",
        "Aktuel kødybde i live-pipelinen (global eller pr. stage)",
        labelnames=("queue",),
        **_gauge_kwargs(),
    )
    feature_compute_ms = Histogram(
        "feature_compute_ms",
        "Feature-beregningstid (ms) per symbol",
        labelnames=("feature", "symbol"),
        buckets=_MS_BUCKETS,
    )
    feature_errors_total = Counter(
        "feature_errors_total",
        "Antal featurefejl (exceptions/NaN) per feature/symbol",
        labelnames=("feature", "symbol"),
    )
    dq_violations_total = Counter(
        "dq_violations_total",
        "Data quality violations by contract and rule",
        labelnames=("contract", "rule"),
    )
    dq_freshness_minutes = Gauge(
        "dq_freshness_minutes",
        "Minutes since last data update",
        labelnames=("dataset",),
        **_gauge_kwargs(),
    )

    _METRICS_READY = True


def bootstrap_core_metrics(venue: str = "binance", symbol: str = "TESTUSDT") -> None:
    """Bootstrapper tomme serier, så /metrics ikke er blankt."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    ensure_registered()

    try:
        feed_transport_latency_ms.labels(venue, symbol).observe(0.0)  # type: ignore[union-attr]
        feed_bar_close_lag_ms.labels(venue, symbol).set(0)  # type: ignore[union-attr]
        feed_bars_total.labels(venue, symbol).inc(0)  # type: ignore[union-attr]
        feed_reconnects_total.labels(venue).inc(0)  # type: ignore[union-attr]
        feed_queue_depth.labels("live").set(0)  # type: ignore[union-attr]
        feature_compute_ms.labels("ema", symbol).observe(0.0)  # type: ignore[union-attr]
        feature_errors_total.labels("ema", symbol).inc(0)  # type: ignore[union-attr]
        dq_violations_total.labels("ohlcv_1h", "bootstrap").inc(0)  # type: ignore[union-attr]
        dq_freshness_minutes.labels("ohlcv_1h").set(0)  # type: ignore[union-attr]
    except Exception:
        pass

    _BOOTSTRAPPED = True


# --- Helper API --------------------------------------------------------------


def observe_transport_latency(venue: str, symbol: str, event_ts_ms: Optional[Union[int, float]]):
    if event_ts_ms is None:
        return
    ensure_registered()
    try:
        d = _now_ms() - _to_ms(event_ts_ms)
        if d >= 0:
            feed_transport_latency_ms.labels(venue, symbol).observe(d)  # type: ignore[union-attr]
    except Exception:
        return


def observe_transport_latency_ms(venue: str, symbol: str, ms: Union[int, float]):
    ensure_registered()
    try:
        d = float(ms)
        if d >= 0:
            feed_transport_latency_ms.labels(venue, symbol).observe(d)  # type: ignore[union-attr]
    except Exception:
        return


def set_bar_close_lag(venue: str, symbol: str, bar_end_ts: Union[int, float]):
    ensure_registered()
    try:
        d = _now_ms() - _to_ms(bar_end_ts)
        if d >= 0:
            feed_bar_close_lag_ms.labels(venue, symbol).set(d)  # type: ignore[union-attr]
    except Exception:
        return


def observe_bar_close_lag_ms(venue: str, symbol: str, lag_ms: Union[int, float]):
    ensure_registered()
    try:
        d = float(lag_ms)
        if d >= 0:
            feed_bar_close_lag_ms.labels(venue, symbol).set(d)  # type: ignore[union-attr]
    except Exception:
        return


def inc_bars(venue: str, symbol: str, n: int = 1):
    ensure_registered()
    if n > 0:
        feed_bars_total.labels(venue, symbol).inc(n)  # type: ignore[union-attr]


def inc_feed_bars_total(venue: str, symbol: str, n: int = 1):
    inc_bars(venue, symbol, n)


def inc_reconnect(venue: str):
    ensure_registered()
    feed_reconnects_total.labels(venue).inc()  # type: ignore[union-attr]


def inc_feed_reconnects_total(venue: str):
    inc_reconnect(venue)


def set_queue_depth(depth: int, queue_name: str = "live"):
    ensure_registered()
    if depth >= 0:
        feed_queue_depth.labels(queue_name).set(depth)  # type: ignore[union-attr]


@contextmanager
def time_feature(feature: str, symbol: str):
    ensure_registered()
    start_ns = time.perf_counter_ns()
    try:
        yield
    except Exception:
        feature_errors_total.labels(feature, symbol).inc()  # type: ignore[union-attr]
        raise
    finally:
        dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
        if dur_ms >= 0:
            feature_compute_ms.labels(feature, symbol).observe(dur_ms)  # type: ignore[union-attr]


# --------- Fase 5 helpers (DQ) ----------------------------------------------


def inc_dq_violation(contract: str, rule: str):
    ensure_registered()
    dq_violations_total.labels(contract, rule).inc()  # type: ignore[union-attr]


def set_dq_freshness_minutes(dataset: str, minutes: Union[int, float]):
    ensure_registered()
    try:
        val = float(minutes)
        if val >= 0:
            dq_freshness_minutes.labels(dataset).set(val)  # type: ignore[union-attr]
    except Exception:
        return


# --------- Dev helpers (emulér trafik) --------------------------------------


def emit_sample(
    venue: str,
    symbol: str = "TESTUSDT",
    transport_ms: Optional[float] = None,
    bar_close_lag_ms: Optional[float] = None,
    bars_inc: int = 1,
) -> dict:
    """
    Emulerer ét "tick" for et givent venue/symbol:
      - inc feed_bars_total
      - observe feed_transport_latency_ms (histogram)
      - set feed_bar_close_lag_ms (gauge)
      - observe feature_compute_ms for 'ema'
    Bruges af debug-rute. Returnerer et status-dict.
    """
    ensure_registered()
    try:
        # bars
        if bars_inc and bars_inc > 0:
            feed_bars_total.labels(venue, symbol).inc(bars_inc)  # type: ignore[union-attr]

        # transport latency
        tms = (
            float(transport_ms)
            if transport_ms is not None
            else random.choice([20, 35, 60, 120, 240, 320])
        )
        feed_transport_latency_ms.labels(venue, symbol).observe(tms)  # type: ignore[union-attr]

        # bar close lag
        bl = (
            float(bar_close_lag_ms)
            if bar_close_lag_ms is not None
            else random.choice([500, 800, 1200])
        )
        feed_bar_close_lag_ms.labels(venue, symbol).set(bl)  # type: ignore[union-attr]

        # feature latency (for dashboards)
        feature_compute_ms.labels("ema", symbol).observe(random.choice([8, 12, 18, 30, 45]))  # type: ignore[union-attr]

        return {
            "ok": True,
            "venue": venue,
            "symbol": symbol,
            "transport_ms": tms,
            "bar_close_lag_ms": bl,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "venue": venue, "symbol": symbol}


def emit_sample_dev() -> dict:
    """
    Bevarer bagudkompatibel "hurtig emulering" for binance/bootstrap.
    """
    try:
        a = emit_sample("binance")
        b = emit_sample("bootstrap")
        return {"ok": True, "results": [a, b]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --- /metrics ASGI app helper -----------------------------------------------


def make_metrics_app():
    """Returnér en ASGI-app for /metrics og bootstrap evt. serier."""
    ensure_registered()
    if _BOOTSTRAP:
        bootstrap_core_metrics()

    if _MULTIPROC:
        from prometheus_client import CollectorRegistry, multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[attr-defined]
        return make_asgi_app(registry=registry)
    return make_asgi_app()


# --- Valgfri FastAPI router til debug-endpoints ------------------------------

_router = None  # lazy-oprettet


def get_debug_router():
    """
    Returnér en FastAPI APIRouter med:
      - POST/GET /_debug/emit_sample?venue=&symbol=&transport_ms=&bar_lag_ms=&n=
    Aktiveres kun hvis ENABLE_DEBUG_ROUTES != 0/false.
    Returnerer None hvis FastAPI ikke er installeret eller ruter er slået fra.
    """
    global _router
    enable = os.getenv("ENABLE_DEBUG_ROUTES", "1").strip().lower() not in {"0", "false"}
    if not enable:
        return None
    if _router is not None:
        return _router

    try:
        from fastapi import APIRouter, Query
    except Exception:
        return None

    ensure_registered()
    r = APIRouter()

    @r.get("/_debug/emit_sample")
    @r.post("/_debug/emit_sample")
    def _emit_sample_route(
        venue: str = Query(default="binance"),
        symbol: str = Query(default="TESTUSDT"),
        transport_ms: Optional[float] = Query(default=None),
        bar_lag_ms: Optional[float] = Query(default=None),
        n: int = Query(default=1, ge=0, le=100),
    ):
        # udfør n gange for at sikre nok samples til rate/window
        out = None
        for _ in range(max(1, n)):
            out = emit_sample(
                venue=venue,
                symbol=symbol,
                transport_ms=transport_ms,
                bar_close_lag_ms=bar_lag_ms,
                bars_inc=1,
            )
        return out or {"ok": False, "error": "emit failed"}

    _router = r
    return _router


__all__ = [
    # metrics
    "feed_transport_latency_ms",
    "feed_bar_close_lag_ms",
    "feed_bars_total",
    "feed_reconnects_total",
    "feed_queue_depth",
    "feature_compute_ms",
    "feature_errors_total",
    "dq_violations_total",
    "dq_freshness_minutes",
    # helpers
    "ensure_registered",
    "bootstrap_core_metrics",
    "observe_transport_latency",
    "observe_transport_latency_ms",
    "set_bar_close_lag",
    "observe_bar_close_lag_ms",
    "inc_bars",
    "inc_feed_bars_total",
    "inc_reconnect",
    "inc_feed_reconnects_total",
    "set_queue_depth",
    "time_feature",
    "inc_dq_violation",
    "set_dq_freshness_minutes",
    "emit_sample",
    "emit_sample_dev",
    "make_metrics_app",
    "get_debug_router",
]

# Auto-init så /metrics viser buckets selv uden live-feed
try:
    if os.getenv("METRICS_AUTO_INIT", "1").strip().lower() not in {"0", "false"}:
        ensure_registered()
        if _BOOTSTRAP:
            bootstrap_core_metrics()
except Exception:
    # Skal aldrig vælte processen
    pass
