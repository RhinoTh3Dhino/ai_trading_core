# bot/live_connector/metrics.py
"""
Prometheus metrics til live-connectoren.

- Metrikkerne er navngivet, så de matcher vores recording rules / dashboards.
- Funktionen `make_metrics_app()` kan bruges til at eksponere /metrics
  (multiprocess-sikkerhed er indbygget, hvis PROMETHEUS_MULTIPROC_DIR er sat).
- Alle helper-funktioner sørger for at ignorere negative værdier og håndtere
  timestamps i både sekunder og millisekunder.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional, Union

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Multiprocess flag: Hvis du senere kører uvicorn med --workers > 1 og sætter
# PROMETHEUS_MULTIPROC_DIR, kan vi vælge fornuftige Gauge-aggregationsmodi.
_MULTIPROC = bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))

# NOTE: ms-buckets der dækker både LAN og spikes
_MS_BUCKETS = (
    1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750,
    1_000, 1_500, 2_000, 3_000, 5_000, 7_500, 10_000
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(ts: Union[int, float]) -> int:
    """
    Konverter et timestamp til millisekunder.
    Hvis tallet ser ud til at være i sekunder (< 1e12), konverter til ms.
    """
    tsf = float(ts)
    if tsf < 1e12:  # sandsynligvis sekunder
        tsf *= 1000.0
    return int(tsf)


def _gauge_kwargs() -> dict:
    """
    Vælg multiprocess-aggregationsmodus for Gauges.
    - 'max' giver mening for lag og kødybde (vi vil typisk se max på tværs af workers).
    """
    return {"multiprocess_mode": "max"} if _MULTIPROC else {}


# --- Core metrics (navne matcher dashboards/alerts/recording rules) ---

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

# (Valgfri) fejlrate til alarmering
feature_errors_total = Counter(
    "feature_errors_total",
    "Antal featurefejl (exceptions/NaN) per feature/symbol",
    labelnames=("feature", "symbol"),
)


# --- Helper API ---

def observe_transport_latency(venue: str, symbol: str, event_ts_ms: Optional[Union[int, float]]):
    """
    Observer transport-latens: now_ms - event_ts_ms.
    'event_ts_ms' må være i ms eller sekunder (detekteres automatisk).
    """
    if event_ts_ms is None:
        return
    try:
        d = _now_ms() - _to_ms(event_ts_ms)
    except Exception:
        return
    if d >= 0:
        feed_transport_latency_ms.labels(venue, symbol).observe(d)


def set_bar_close_lag(venue: str, symbol: str, bar_end_ts: Union[int, float]):
    """
    Sæt lag for bar close: now_ms - bar_end_ms.
    'bar_end_ts' må være i ms eller sekunder.
    """
    try:
        d = _now_ms() - _to_ms(bar_end_ts)
    except Exception:
        return
    if d >= 0:
        feed_bar_close_lag_ms.labels(venue, symbol).set(d)


def inc_bars(venue: str, symbol: str, n: int = 1):
    if n <= 0:
        return
    feed_bars_total.labels(venue, symbol).inc(n)


def inc_reconnect(venue: str):
    feed_reconnects_total.labels(venue).inc()


def set_queue_depth(depth: int, queue_name: str = "live"):
    """
    Sæt kødybde (negativt ignoreres).
    """
    if depth < 0:
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
    start_ns = time.perf_counter_ns()
    try:
        yield
    except Exception:
        feature_errors_total.labels(feature, symbol).inc()
        raise
    finally:
        dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
        if dur_ms >= 0:
            feature_compute_ms.labels(feature, symbol).observe(dur_ms)


# --- /metrics ASGI app helper (valgfrit) ---

def make_metrics_app():
    """
    Returnér en ASGI-app for /metrics.
    - Hvis PROMETHEUS_MULTIPROC_DIR er sat, bygger vi en multiprocess-sikker app.
    - Ellers bruger vi standard-registry.
    Brug i runner:
        from bot.live_connector.metrics import make_metrics_app
        app.mount("/metrics", make_metrics_app())
    """
    if _MULTIPROC:
        # Multiprocess-safe registry
        from prometheus_client import CollectorRegistry, multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[attr-defined]
        return make_asgi_app(registry=registry)
    # Single-process (nuværende setup)
    return make_asgi_app()


__all__ = [
    # metrics
    "feed_transport_latency_ms",
    "feed_bar_close_lag_ms",
    "feed_bars_total",
    "feed_reconnects_total",
    "feed_queue_depth",
    "feature_compute_ms",
    "feature_errors_total",
    # helpers
    "observe_transport_latency",
    "set_bar_close_lag",
    "inc_bars",
    "inc_reconnect",
    "set_queue_depth",
    "time_feature",
    "make_metrics_app",
]
