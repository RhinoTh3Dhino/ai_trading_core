# tests/test_metrics_exposition.py
import asyncio
import subprocess
import time

import httpx

# Forudsætter at uvicorn kører app på :8000 i CI-jobbet


def test_metrics_endpoint_has_core_metrics():
    r = httpx.get("http://localhost:8000/metrics", timeout=5)
    body = r.text
    assert "feed_transport_latency_ms_bucket" in body
    assert "feed_bar_close_lag_ms" in body
    assert "feed_bars_total" in body
    assert "feed_reconnects_total" in body
    assert "feed_queue_depth" in body
    assert "feature_compute_ms_bucket" in body
