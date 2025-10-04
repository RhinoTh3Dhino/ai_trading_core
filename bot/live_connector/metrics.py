# bot/live_connector/metrics.py
"""
Prometheus metrics til live-connectoren.

- Navne matcher recording rules / dashboards.
- Idempotent registrering via ensure_registered() (kan kaldes flere gange).
- make_metrics_app() kalder ensure_registered() og bootstrapper som standard,
  så /metrics eksponerer histogram/gauge/counter-serierne – også uden live-feed.
- Helper-funktioner kalder defensivt ensure_registered() for at undgå race.

NYT:
- Auto-init ved import (kan slås fra med METRICS_AUTO_INIT=0) så *_bucket-linjer
  altid er synlige i /metrics – også hvis en anden app allerede eksponerer endpointet.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional, Union

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Multiprocess flag: Hvis du kører uvicorn med --workers>1 og sætter
# PROMETHEUS_MULTIPROC_DIR, vælger vi passende aggregationsmodus for Gauges.
_MULTIPROC = bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))

# Styr om vi bootstrapper en "tom" serie pr. metric ved app-start
# (så histogram-buckets m.m. altid er synlige i /metrics)
_BOOTSTRAP = (os.getenv("METRICS_BOOTSTRAP", "1").strip().lower() not in {"0", "false"})

# ms-buckets der dækker både lav latenstid og spikes
_MS_BUCKETS = (
    1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750,
    1_000, 1_500, 2_000, 3_000, 5_000, 7_500, 10_000
)

# Globals (sættes ved ensure_registered)
feed_transport_latency_ms: Histogram | None = None
feed_bar_close_lag_ms: Gauge | None = None
feed_bars_total: Counter | None = None
feed_reconnects_total: Counter | None = None
feed_queue_depth: Gauge | None = None
feature_compute_ms: Histogram | None = None
feature_errors_total: Counter | None = None

_METRICS_READY = False
_BOOTSTRAPPED = False  # sikrer, at vi kun bootstrapper én gang


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(ts: Union[int, float]) -> int:
    """
    Konverter timestamp til ms. Hvis <1e12, antag sekunder og skaler til ms.
    """
    tsf = float(ts)
    if tsf < 1e12:
        tsf *= 1000.0
    return int(tsf)


def _gauge_kwargs() -> dict:
    """
    Multiprocess-aggregationsmodus for Gauges.
    'max' giver mening for lag/kødybde.
    """
    return {"multiprocess_mode": "max"} if _MULTIPROC else {}


def ensure_registered() -> None:
    """
    Opret (én gang) alle metrikker. Sikker at kalde flere gange.
    Løser bl.a. problemet hvor /metrics ellers ikke viser *_bucket-linjer.
    """
    global _METRICS_READY
    if _METRICS_READY:
        return

    global feed_transport_latency_ms, feed_bar_close_lag_ms, feed_bars_total
    global feed_reconnects_total, feed_queue_depth, feature_compute_ms, feature_errors_total

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

    _METRICS_READY = True


def bootstrap_core_metrics(venue: str = "binance", symbol: str = "TESTUSDT") -> None:
    """
    Opretter mindst én label-child pr. metric med en 0-observation/0-sæt,
    så serierne altid er synlige på /metrics før første rigtige datapunkt.

    Bruges i make_metrics_app() ved startup (kan slås fra med METRICS_BOOTSTRAP=0).
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    ensure_registered()

    try:
        if feed_transport_latency_ms is not None:
            feed_transport_latency_ms.labels(venue, symbol).observe(0.0)
        if feed_bar_close_lag_ms is not None:
            feed_bar_close_lag_ms.labels(venue, symbol).set(0)
        if feed_bars_total is not None:
            feed_bars_total.labels(venue, symbol).inc(0)
        if feed_reconnects_total is not None:
            feed_reconnects_total.labels(venue).inc(0)
        if feed_queue_depth is not None:
            feed_queue_depth.labels("live").set(0)
        if feature_compute_ms is not None:
            feature_compute_ms.labels("ema", symbol).observe(0.0)
        if feature_errors_total is not None:
            feature_errors_total.labels("ema", symbol).inc(0)
    except Exception:
        # Bootstrap må aldrig vælte processen – det er kun kosmetisk for /metrics
        pass

    _BOOTSTRAPPED = True


# --- Helper API (kalder defensivt ensure_registered) -------------------------

def observe_transport_latency(venue: str, symbol: str, event_ts_ms: Optional[Union[int, float]]):
    """
    Observer transport-latens: now_ms - event_ts_ms (ms eller sekunder).
    """
    if event_ts_ms is None:
        return
    ensure_registered()
    try:
        d = _now_ms() - _to_ms(event_ts_ms)
    except Exception:
        return
    if d >= 0 and feed_transport_latency_ms is not None:
        feed_transport_latency_ms.labels(venue, symbol).observe(d)


def set_bar_close_lag(venue: str, symbol: str, bar_end_ts: Union[int, float]):
    """
    Sæt lag for bar close: now_ms - bar_end_ms (ms eller sekunder).
    """
    ensure_registered()
    try:
        d = _now_ms() - _to_ms(bar_end_ts)
    except Exception:
        return
    if d >= 0 and feed_bar_close_lag_ms is not None:
        feed_bar_close_lag_ms.labels(venue, symbol).set(d)


def inc_bars(venue: str, symbol: str, n: int = 1):
    ensure_registered()
    if n <= 0 or feed_bars_total is None:
        return
    feed_bars_total.labels(venue, symbol).inc(n)


def inc_reconnect(venue: str):
    ensure_registered()
    if feed_reconnects_total is None:
        return
    feed_reconnects_total.labels(venue).inc()


def set_queue_depth(depth: int, queue_name: str = "live"):
    """
    Sæt kødybde (negativt ignoreres).
    """
    ensure_registered()
    if depth < 0 or feed_queue_depth is None:
        return
    feed_queue_depth.labels(queue_name).set(depth)


@contextmanager
def time_feature(feature: str, symbol: str):
    """
    Mål execution-tid for en feature. Inkrementerer også fejl-counter ved exception.
    Brug:
        with time_feature("rsi", "BTCUSDT"):
            compute_rsi(...)
    """
    ensure_registered()
    start_ns = time.perf_counter_ns()
    try:
        yield
    except Exception:
        if feature_errors_total is not None:
            feature_errors_total.labels(feature, symbol).inc()
        raise
    finally:
        dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
        if dur_ms >= 0 and feature_compute_ms is not None:
            feature_compute_ms.labels(feature, symbol).observe(dur_ms)


# --- /metrics ASGI app helper -----------------------------------------------

def make_metrics_app():
    """
    Returnér en ASGI-app for /metrics.
    - Hvis PROMETHEUS_MULTIPROC_DIR er sat, bygger vi en multiprocess-sikker app.
    - Ellers bruger vi standard-registry.
    Sikrer registrering af metrikker først og bootstrapper (med mindre slået fra).
    """
    ensure_registered()
    if _BOOTSTRAP:
        bootstrap_core_metrics()

    if _MULTIPROC:
        from prometheus_client import CollectorRegistry, multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[attr-defined]
        return make_asgi_app(registry=registry)
    return make_asgi_app()


__all__ = [
    # metrics (objekterne kan være None før ensure_registered, så kald funktionen først)
    "feed_transport_latency_ms",
    "feed_bar_close_lag_ms",
    "feed_bars_total",
    "feed_reconnects_total",
    "feed_queue_depth",
    "feature_compute_ms",
    "feature_errors_total",
    # helpers
    "ensure_registered",
    "bootstrap_core_metrics",
    "observe_transport_latency",
    "set_bar_close_lag",
    "inc_bars",
    "inc_reconnect",
    "set_queue_depth",
    "time_feature",
    "make_metrics_app",
]

# --- Auto-init ved import (så buckets vises selv uden live-feed) -----------
try:
    # Slå fra med METRICS_AUTO_INIT=0
    if os.getenv("METRICS_AUTO_INIT", "1").strip().lower() not in {"0", "false"}:
        ensure_registered()
        if _BOOTSTRAP:
            bootstrap_core_metrics()
except Exception:
    # Må aldrig vælte processen – dette er kun for at sikre synlige serier
    pass
