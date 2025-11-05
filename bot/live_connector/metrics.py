# bot/live_connector/metrics.py
"""
Prometheus metrics til live-connectoren.

- Navne matcher recording rules / dashboards.
- Idempotent og reload-sikker registrering via ensure_registered().
- make_metrics_app() kalder ensure_registered() og bootstrapper som standard,
  så /metrics eksponerer histogram/gauge/counter-serier – også uden live-feed.
- Helper-funktioner kalder defensivt ensure_registered() for at undgå race.

NYT (Fase 3):
- Reload-sikkerhed: Ved modul-reload rebindes globale refs til eksisterende
  collectors i default REGISTRY (undgår "Duplicated timeseries").
- Alias-navne til connector-koden:
    inc_feed_bars_total, inc_feed_reconnects_total,
    observe_bar_close_lag_ms, observe_transport_latency_ms
- Auto-init ved import (kan slås fra med METRICS_AUTO_INIT=0).

NYT (Fase 5 — Datakvalitet & alerts):
- dq_violations_total(contract, rule)  [Counter]
- dq_freshness_minutes(dataset)        [Gauge, multiprocess_mode=max]
- Helpers: inc_dq_violation(), set_dq_freshness_minutes()

NYT (Dev/emulering):
- emit_sample_dev(): lille helper til at emulere bars/latency/feature-latency i dev,
  så PromQL-vinduer over 5m får rigtige observationer (ikke tomme sæt).
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, make_asgi_app

# Multiprocess flag: Hvis du kører uvicorn med --workers>1 og sætter
# PROMETHEUS_MULTIPROC_DIR, vælger vi passende aggregationsmodus for Gauges.
_MULTIPROC = bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))

# Styr om vi bootstrapper en "tom" serie pr. metric ved app-start
# (så histogram-buckets m.m. altid er synlige i /metrics)
_BOOTSTRAP = os.getenv("METRICS_BOOTSTRAP", "1").strip().lower() not in {"0", "false"}

# ms-buckets der dækker både lav latenstid og spikes
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
    'max' giver mening for lag/kødybde/freshness.
    """
    return {"multiprocess_mode": "max"} if _MULTIPROC else {}


def _registry_lookup(name: str) -> Any | None:
    """
    Find en eksisterende collector i default REGISTRY for metric-navnet `name`.
    Bruger REGISTRY._names_to_collectors (privat API; ok for intern robusthed/tests).
    """
    try:
        mapping: Dict[str, Any] = getattr(REGISTRY, "_names_to_collectors", {})  # type: ignore[attr-defined]
        return mapping.get(name)
    except Exception:
        return None


def ensure_registered() -> None:
    """
    Opret (én gang) alle metrikker. Sikker at kalde flere gange og reload-sikker.
    Hvis collectors allerede ligger i REGISTRY (fra tidligere import), rebind globals.
    """
    global _METRICS_READY
    if _METRICS_READY:
        return

    global feed_transport_latency_ms, feed_bar_close_lag_ms, feed_bars_total
    global feed_reconnects_total, feed_queue_depth, feature_compute_ms, feature_errors_total
    global dq_violations_total, dq_freshness_minutes

    # 1) Rebind til eksisterende collectors hvis de findes (reload-sikkert)
    existing_transport = _registry_lookup("feed_transport_latency_ms")
    existing_bar_lag = _registry_lookup("feed_bar_close_lag_ms")
    existing_bars = _registry_lookup("feed_bars_total")
    existing_reconnects = _registry_lookup("feed_reconnects_total")
    existing_queue = _registry_lookup("feed_queue_depth")
    existing_feat_ms = _registry_lookup("feature_compute_ms")
    existing_feat_err = _registry_lookup("feature_errors_total")
    existing_dq_viol = _registry_lookup("dq_violations_total")
    existing_dq_fresh = _registry_lookup("dq_freshness_minutes")

    if all(
        [
            existing_transport,
            existing_bar_lag,
            existing_bars,
            existing_reconnects,
            existing_queue,
            existing_feat_ms,
            existing_feat_err,
            existing_dq_viol is not None,
            existing_dq_fresh is not None,
        ]
    ):
        feed_transport_latency_ms = existing_transport
        feed_bar_close_lag_ms = existing_bar_lag
        feed_bars_total = existing_bars
        feed_reconnects_total = existing_reconnects
        feed_queue_depth = existing_queue
        feature_compute_ms = existing_feat_ms
        feature_errors_total = existing_feat_err
        dq_violations_total = existing_dq_viol
        dq_freshness_minutes = existing_dq_fresh
        _METRICS_READY = True
        return

    # 2) Ellers registrer dem (første gang i processen)
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

    # Bemærk: queue-depth holdes global/pr-stage (ikke per venue)
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

    # ---- Fase 5: Data Quality ----
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

        # Fase 5 bootstrap
        if dq_violations_total is not None:
            dq_violations_total.labels("ohlcv_1h", "bootstrap").inc(0)
        if dq_freshness_minutes is not None:
            dq_freshness_minutes.labels("ohlcv_1h").set(0)
    except Exception:
        # Bootstrap må aldrig vælte processen – det er kun kosmetisk for /metrics
        pass

    _BOOTSTRAPPED = True


# --- Helper API (kalder defensivt ensure_registered) -------------------------


def observe_transport_latency(venue: str, symbol: str, event_ts_ms: Optional[Union[int, float]]):
    """
    Observer transport-latens fra event_ts (ms eller sekunder): now_ms - event_ts_ms.
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


def observe_transport_latency_ms(venue: str, symbol: str, ms: Union[int, float]):
    """
    Observer transport-latens når du allerede har ms-differensen.
    """
    ensure_registered()
    try:
        d = float(ms)
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


def observe_bar_close_lag_ms(venue: str, symbol: str, lag_ms: Union[int, float]):
    """
    Sæt bar-close lag direkte i millisekunder.
    """
    ensure_registered()
    try:
        d = float(lag_ms)
    except Exception:
        return
    if d >= 0 and feed_bar_close_lag_ms is not None:
        feed_bar_close_lag_ms.labels(venue, symbol).set(d)


def inc_bars(venue: str, symbol: str, n: int = 1):
    ensure_registered()
    if n <= 0 or feed_bars_total is None:
        return
    feed_bars_total.labels(venue, symbol).inc(n)


def inc_feed_bars_total(venue: str, symbol: str, n: int = 1):
    """
    Alias for inc_bars – brugt af connector-koden.
    """
    inc_bars(venue, symbol, n)


def inc_reconnect(venue: str):
    ensure_registered()
    if feed_reconnects_total is None:
        return
    feed_reconnects_total.labels(venue).inc()


def inc_feed_reconnects_total(venue: str):
    """
    Alias for inc_reconnect – brugt af connector-koden.
    """
    inc_reconnect(venue)


def set_queue_depth(depth: int, queue_name: str = "live"):
    """
    Sæt global/pr-stage kødybde (negativt ignoreres).
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


# --------- Fase 5 helpers (DQ) ----------------------------------------------


def inc_dq_violation(contract: str, rule: str):
    """
    Inkrementér en DQ-violation (bruges når validate(...) returnerer issues).
    """
    ensure_registered()
    if dq_violations_total is None:
        return
    dq_violations_total.labels(contract, rule).inc()


def set_dq_freshness_minutes(dataset: str, minutes: Union[int, float]):
    """
    Sæt freshness (minutter siden sidste opdatering) for et dataset.
    Typisk kaldt fra persistens-job efter succesfuld write/rotation.
    """
    ensure_registered()
    if dq_freshness_minutes is None:
        return
    try:
        val = float(minutes)
    except Exception:
        return
    if val >= 0:
        dq_freshness_minutes.labels(dataset).set(val)


# --------- Dev helper: emulér trafik til tests/gates -------------------------


def emit_sample_dev() -> dict:
    """
    Emulér et enkelt “tick” for at sikre aktive tidsserier i Prometheus.
    Kald denne fra en debug-route (f.eks. /_debug/emit_sample).

    Returnerer et lille statusdict som kan logges i debug.
    """
    ensure_registered()

    venues = ("binance", "bootstrap")
    symbol = "TESTUSDT"

    try:
        for v in venues:
            # Bars
            if feed_bars_total is not None:
                feed_bars_total.labels(v, symbol).inc()

            # Transport-latency (histogram observationer)
            if feed_transport_latency_ms is not None:
                feed_transport_latency_ms.labels(v, symbol).observe(
                    random.choice([20, 35, 60, 120, 240])
                )

            # Bar close lag (gauge)
            if feed_bar_close_lag_ms is not None:
                feed_bar_close_lag_ms.labels(v, symbol).set(random.choice([500, 800, 1200]))

        # Feature compute latency (histogram observationer)
        if feature_compute_ms is not None:
            feature_compute_ms.labels("ema", symbol).observe(random.choice([8, 12, 18, 30, 45]))

        # Kødybde (gauge)
        if feed_queue_depth is not None:
            feed_queue_depth.labels("live").set(random.choice([0, 5, 10, 20]))

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
    # metrics (objekterne kan være None før ensure_registered)
    "feed_transport_latency_ms",
    "feed_bar_close_lag_ms",
    "feed_bars_total",
    "feed_reconnects_total",
    "feed_queue_depth",
    "feature_compute_ms",
    "feature_errors_total",
    "dq_violations_total",
    "dq_freshness_minutes",
    # helpers (primær + alias)
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
    # Fase 5 helpers
    "inc_dq_violation",
    "set_dq_freshness_minutes",
    # Dev helper
    "emit_sample_dev",
    "make_metrics_app",
]

# --- Auto-init ved import (så buckets vises selv uden live-feed) ------------
try:
    # Slå fra med METRICS_AUTO_INIT=0
    if os.getenv("METRICS_AUTO_INIT", "1").strip().lower() not in {"0", "false"}:
        ensure_registered()
        if _BOOTSTRAP:
            bootstrap_core_metrics()
except Exception:
    # Må aldrig vælte processen – dette er kun for at sikre synlige serier
    pass
