# bot/metrics_core.py
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# Brug modul-scope variabler så de kun konstrueres én gang pr. process.
# Hver konstruktion er idempotent pga. Python-import cache; ved genkonstruktion
# i samme registry ville prometheus_client ellers kaste ValueError.

# Histogrammer → giver *_bucket linjer
_FEED_TRANSPORT = Histogram(
    "feed_transport_latency_ms",
    "Transport latency from feed to pipeline (ms)",
    registry=REGISTRY,
)
_FEATURE_COMPUTE = Histogram(
    "feature_compute_ms",
    "Feature computation time (ms)",
    registry=REGISTRY,
)

# Gauges
_FEED_BAR_CLOSE_LAG = Gauge(
    "feed_bar_close_lag_ms",
    "Lag between bar close and processing (ms)",
    registry=REGISTRY,
)
_FEED_QUEUE_DEPTH = Gauge(
    "feed_queue_depth",
    "Current depth of feed queue",
    registry=REGISTRY,
)

# Counters
_FEED_BARS_TOTAL = Counter(
    "feed_bars_total",
    "Total bars processed",
    registry=REGISTRY,
)
_FEED_RECONNECTS_TOTAL = Counter(
    "feed_reconnects_total",
    "Total reconnects to feed",
    registry=REGISTRY,
)


def init_core_metrics() -> None:
    """
    NOP-funktion – selve importen sørger for registreringen.
    Bevidst tom så du trygt kan kalde den flere gange.
    """
    return None
