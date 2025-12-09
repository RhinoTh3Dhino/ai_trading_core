# bot/live_connector/engine.py
from __future__ import annotations

import logging
import os

import uvicorn

from .config import LiveConfig

logger = logging.getLogger(__name__)


def run_live_connector(cfg: LiveConfig) -> None:
    """
    Central entrypoint for live-loop.

    Her bevarer vi eksisterende adfærd:
    - Starter FastAPI-app'en `bot.live_connector.runner:app` via uvicorn.
    - Benytter port 8000 (eller LIVE_API_PORT, hvis sat).
    - Logger den konfiguration, CLI'en har parsed.
    """

    api_host = os.getenv("LIVE_API_HOST", "0.0.0.0")
    api_port = int(os.getenv("LIVE_API_PORT", "8000"))

    logger.info(
        "Starter live-connector: host=%s port=%s venues=%s symbols=%s interval=%s "
        "quiet=%s status_min_secs=%s partitioning_enabled=%s output_root=%s",
        api_host,
        api_port,
        cfg.venues,
        cfg.symbols,
        cfg.interval,
        cfg.quiet,
        cfg.status_min_secs,
        cfg.partitioning_enabled,
        cfg.output_root,
    )

    # OBS: selve app-logikken ligger stadig i bot.live_connector.runner:app
    uvicorn.run(
        "bot.live_connector.runner:app",
        host=api_host,
        port=api_port,
        workers=1,
    )
