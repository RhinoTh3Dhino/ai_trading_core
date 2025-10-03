# bot/live_connector/metrics_bootstrap.py
from prometheus_client import Histogram, Counter, Gauge

# === Feed / transport ===
FEED_TRANSPORT_LATENCY_MS = Histogram(
    "feed_transport_latency_ms",
    "End-to-end transportlatens (ms): now_ms - event_ts fra venue besked",
    ["venue"],
    # simple buckets – tilpas hvis du vil
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
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
    buckets=(1, 2, 5, 10, 20, 50, 100, 250, 500)
)

FEATURE_ERRORS_TOTAL = Counter(
    "feature_errors_total",
    "Antal featurefejl (exceptions/NaN) per feature/symbol",
    ["feature", "symbol"],
)
