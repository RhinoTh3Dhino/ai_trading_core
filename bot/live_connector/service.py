# bot/live_connector/service.py
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import yaml
from fastapi import FastAPI, Header, HTTPException, Query
from prometheus_client import start_http_server

# Lokal imports: vi forventer at metrics.py definerer disse
# (fail-safe: vi bruger getattr for at undgå import-fejl i edge cases)
from . import metrics as lc_metrics  # type: ignore
from .venues.okx import OKXConnector
from .ws_client import WSClient


# ----------------------- konfig / utils -----------------------
def load_yaml(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def on_kline(evt: dict):
    # TODO: send til event-bus/pipeline. Midlertidigt: skriv til log/STDOUT
    print(json.dumps(evt, separators=(",", ":")))


def _env_true(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


# ----------------------- FastAPI app-factory -----------------------
def create_app() -> FastAPI:
    """
    Opretter en FastAPI-app uden side-effekter (ingen auto-start af server).
    Runner/uvicorn importerer denne som `bot.live_connector.runner:app`.
    """
    app = FastAPI(title="live-connector", version="1.0")

    ENABLE_DEBUG_ROUTES = _env_true("ENABLE_DEBUG_ROUTES", "0")
    DQ_SHARED_SECRET = os.getenv("DQ_SHARED_SECRET", "").strip()

    # Health/ready probes (enkle, men nok til CI/compose)
    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        return {"ok": True}

    # Debug-route til at prime metrics (kun hvis eksplicit slået til)
    @app.post("/_debug/emit_sample")
    async def emit_sample_debug():
        if not ENABLE_DEBUG_ROUTES:
            # Matcher CI-fejlen hvor der kom 403 – nu kontrolleret & intentionel
            raise HTTPException(status_code=403, detail="Debug routes disabled")

        # Kald en lille helper hvis den findes (idempotent)
        prime = getattr(lc_metrics, "prime_debug_sample", None)
        if callable(prime):
            prime()
        else:
            # Fallback: prøv at skrive lidt til gængse metrics, hvis de findes
            _observe_if_exists("FEED_TRANSPORT_LATENCY_MS", 123.0)
            _inc_if_exists("FEATURE_ERRORS_TOTAL", labels={"feature": "ema"})
        return {"ok": True}

    # Prod-sikker endpoint til at opdatere DQ freshness
    @app.post("/dq/freshness")
    async def dq_freshness(
        dataset: str = Query(..., min_length=1),
        minutes: float = Query(1.0, ge=0.0),
        x_dq_secret: Optional[str] = Header(default=None, alias="X-Dq-Secret"),
    ):
        # Hvis der er sat secret i miljøet, håndhæv den
        if DQ_SHARED_SECRET and (x_dq_secret or "") != DQ_SHARED_SECRET:
            raise HTTPException(status_code=403, detail="Forbidden")

        # Brug officiel helper hvis den er tilgængelig
        set_freshness = getattr(lc_metrics, "set_freshness", None)
        gauge = getattr(lc_metrics, "DQ_FRESHNESS_MINUTES", None)

        if callable(set_freshness):
            set_freshness(dataset, minutes)
        elif gauge is not None:
            # labels() sikrer en serie pr. dataset
            try:
                gauge.labels(dataset=dataset).set(minutes)
            except Exception as e:  # pragma: no cover (defensivt)
                raise HTTPException(status_code=500, detail=f"gauge set failed: {e}")
        else:
            # Hvis intet eksisterer, giv en pæn fejl – hjælper ved fejlopsætning
            raise HTTPException(status_code=500, detail="DQ_FRESHNESS_MINUTES not available")

        return {"ok": True, "dataset": dataset, "minutes": minutes}

    return app


# Hjælpe-funktioner til defensiv metrics-priming
def _observe_if_exists(histogram_name: str, value: float):
    try:
        h = getattr(lc_metrics, histogram_name, None)
        if h is not None:
            # many histograms er uden labels; hvis labels findes, brug default
            h.observe(value)  # type: ignore[attr-defined]
    except Exception:
        pass  # defensivt: ingen hard-fail i debug helper


def _inc_if_exists(counter_name: str, labels: Optional[dict] = None):
    try:
        c = getattr(lc_metrics, counter_name, None)
        if c is not None:
            if labels:
                c.labels(**labels).inc()  # type: ignore[attr-defined]
            else:
                c.inc()  # type: ignore[attr-defined]
    except Exception:
        pass


# ----------------------- CLI entry (OKX connector) -----------------------
async def _main_connector():
    cfg = load_yaml("config/venue_okx.yaml")
    smap = load_yaml("config/symbol_map.yaml")
    ws = WSClient()
    c = OKXConnector(cfg, smap, on_kline, ws)
    await c.run()


if __name__ == "__main__":
    # Lokal kørsel af connector + Prometheus /metrics på :9000
    # (runner/uvicorn i CI bruger create_app() og eksponerer /metrics via egen stack)
    start_http_server(9000)
    asyncio.run(_main_connector())
