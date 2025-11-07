# bot/live_connector/metrics_bootstrap.py
"""
Prometheus metrics for live-connector.

Idempotent by design: hvis modulet genimporteres i samme proces, redefinerer vi
ikke collectors (for at undgå "Duplicated timeseries in CollectorRegistry").
"""

from __future__ import annotations

from typing import Dict

from prometheus_client import Counter, Gauge, Histogram

# ----------------------- Idempotent guards -----------------------
# Hvis globals allerede indeholder collector-objekterne, så brug dem.
# Det er en simpel, men effektiv beskyttelse imod utilsigtet re-import.
if "FEED_TRANSPORT_LATENCY_MS" not in globals():
    # === Feed / transport ===
    FEED_TRANSPORT_LATENCY_MS = Histogram(
        "feed_transport_latency_ms",
        "End-to-end transportlatens (ms): now_ms - event_ts fra venue besked",
        ["venue"],
        buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
    )

    FEED_BAR_CLOSE_LAG_MS = Gauge(
        "feed_bar_close_lag_ms",
        "Hvor langt inde i baren vi er: now_ms - bar_end_ms",
        ["venue", "symbol"],
    )

    FEED_BARS_TOTAL = Counter(
        "feed_bars_total",
        "Antal lukkede bars (is_final=True) modtaget/produceret",
        ["venue"],
    )

    FEED_RECONNECTS_TOTAL = Counter(
        "feed_reconnects_total",
        "Antal WS reconnects per venue",
        ["venue"],
    )

    FEED_QUEUE_DEPTH = Gauge(
        "feed_queue_depth",
        "Aktuel kødybde i live-pipelinen (global eller pr. stage)",
        ["queue"],
    )

    # === Features ===
    FEATURE_COMPUTE_MS = Histogram(
        "feature_compute_ms",
        "Feature-beregningstid (ms) per symbol",
        ["feature", "symbol"],
        buckets=(1, 2, 5, 10, 20, 50, 100, 250, 500),
    )

    FEATURE_ERRORS_TOTAL = Counter(
        "feature_errors_total",
        "Antal featurefejl (exceptions/NaN) per feature/symbol",
        ["feature", "symbol"],
    )

    # === Data Quality (DQ) ===
    # Matcher alarmer/queries i ops/prometheus/*
    DQ_FRESHNESS_MINUTES = Gauge(
        "dq_freshness_minutes",
        "Hvor mange minutter siden seneste vellykkede ingestion (per dataset).",
        ["dataset"],
    )


# ----------------------- Helpers (bruges af service.py) -----------------------
def prime_debug_sample() -> None:
    """
    Laver en minimal, ufarlig priming af metrics så histogram-buckets materialiseres
    og smoke-tests kan læse noget meningsfuldt fra /metrics.
    """
    try:
        FEED_TRANSPORT_LATENCY_MS.labels(venue="binance").observe(123.0)
    except Exception:
        pass
    try:
        FEATURE_COMPUTE_MS.labels(feature="ema", symbol="BTCUSDT").observe(7.0)
    except Exception:
        pass
    try:
        FEATURE_ERRORS_TOTAL.labels(feature="ema", symbol="BTCUSDT").inc()
    except Exception:
        pass
    try:
        DQ_FRESHNESS_MINUTES.labels(dataset="ohlcv_1h").set(1.0)
    except Exception:
        pass


def set_freshness(dataset: str, minutes: float) -> None:
    """
    Sæt dq_freshness_minutes{dataset} = minutes.
    Bruges af /dq/freshness endpointet.
    """
    DQ_FRESHNESS_MINUTES.labels(dataset=dataset).set(float(minutes))


__all__ = [
    # Metrics
    "FEED_TRANSPORT_LATENCY_MS",
    "FEED_BAR_CLOSE_LAG_MS",
    "FEED_BARS_TOTAL",
    "FEED_RECONNECTS_TOTAL",
    "FEED_QUEUE_DEPTH",
    "FEATURE_COMPUTE_MS",
    "FEATURE_ERRORS_TOTAL",
    "DQ_FRESHNESS_MINUTES",
    # Helpers
    "prime_debug_sample",
    "set_freshness",
]
