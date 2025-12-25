# bot/live_connector/engine.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import uvicorn
import yaml

from bot.strategies.flagship_trend_v1 import FlagshipTrendConfig, FlagshipTrendV1Strategy

from .config import LiveConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy registry + loader
# ---------------------------------------------------------------------------


@dataclass
class StrategyBootstrap:
    """
    Holder metadata om den aktive strategi, så vi kan logge og inspicere den.
    """

    id: str
    instance: Any
    config: Any


# Kendte strategier i live_connector
STRATEGY_REGISTRY: Dict[str, Tuple[Type[Any], Type[Any]]] = {
    "flagship_trend_v1": (FlagshipTrendV1Strategy, FlagshipTrendConfig),
}


def load_strategy_from_env() -> StrategyBootstrap:
    """
    Loader strategi ud fra:
      - env-var LYRA_STRATEGY_ID (default: 'flagship_trend_v1')
      - YAML-config i config/strategies/{id}.yml

    Forventet YAML-struktur:

    id: flagship_trend_v1
    params:
      symbol: "BTCUSDT"
      timeframe: "1h"
      ...

    hvor 'params' matcher felterne i FlagshipTrendConfig.
    """
    strategy_id = os.getenv("LYRA_STRATEGY_ID", "flagship_trend_v1")

    if strategy_id not in STRATEGY_REGISTRY:
        raise RuntimeError(
            f"Ukendt LYRA_STRATEGY_ID='{strategy_id}'. "
            f"Kendte strategier: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_cls, cfg_cls = STRATEGY_REGISTRY[strategy_id]

    cfg_path = Path("config") / "strategies" / f"{strategy_id}.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Strategi-config mangler: {cfg_path}. "
            "Opret filen baseret på flagship_trend_v1.yml-skabelonen."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Ugyldigt YAML-indhold i {cfg_path}: forventet mapping.")

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"Ugyldig struktur i {cfg_path}: 'params' skal være en mapping.")

    cfg = cfg_cls(**params)
    strategy = strategy_cls(cfg)

    logger.info(
        "Strategy bootstrap: id=%s cfg_path=%s params_keys=%s",
        strategy_id,
        cfg_path,
        sorted(params.keys()),
    )

    return StrategyBootstrap(id=strategy_id, instance=strategy, config=cfg)


# ---------------------------------------------------------------------------
# LiveEngine – tynd wrapper omkring strategi, som runner/service kan bruge
# ---------------------------------------------------------------------------


class LiveEngine:
    """
    Binder live_connector sammen med Flagship-strategien.

    Selve FastAPI/WS-logikken ligger fortsat i runner.py/service.py.
    Denne klasse kan bruges derfra til at kalde strategy.on_bar(...) osv.
    """

    def __init__(self, live_cfg: LiveConfig) -> None:
        self.live_cfg = live_cfg

        # Loader strategi fra env + YAML
        sb = load_strategy_from_env()
        self.strategy_id: str = sb.id
        self.strategy = sb.instance
        self.strategy_config = sb.config

        logger.info(
            "LiveEngine initialiseret med strategi=%s timeframe=%s symbols=%s",
            self.strategy_id,
            getattr(self.strategy_config, "timeframe", None),
            getattr(self.strategy_config, "symbol", None),
        )

    def on_bar(self, bar: Any, account_state: Any, daily_pnl: float):
        """
        Delegér til strategiens on_bar.

        Denne metode forventes kaldt fra runner/service, når der kommer
        en ny bar fra feed'et.

        Returnerer et Signal-objekt eller None, afhængigt af strategi-logikken.
        """
        return self.strategy.on_bar(
            bar=bar,
            account_state=account_state,
            daily_pnl=daily_pnl,
        )


# ---------------------------------------------------------------------------
# Eksisterende entrypoint til uvicorn – uændret adfærd
# ---------------------------------------------------------------------------


def run_live_connector(cfg: LiveConfig) -> None:
    """
    Central entrypoint for live-loop.

    Bevarer eksisterende adfærd:
    - Starter FastAPI-app'en `bot.live_connector.runner:app` via uvicorn.
    - Benytter port 8000 (eller LIVE_API_PORT, hvis sat).
    - Logger den konfiguration, CLI'en har parsed.

    Strategy-initialisering håndteres separat via LiveEngine/strategy-loader
    og kan bruges fra runner/service efter behov.
    """

    api_host = os.getenv("LIVE_API_HOST", "0.0.0.0")
    api_port = int(os.getenv("LIVE_API_PORT", "8000"))
    strategy_id = os.getenv("LYRA_STRATEGY_ID", "flagship_trend_v1")

    logger.info(
        "Starter live-connector: host=%s port=%s venues=%s symbols=%s interval=%s "
        "quiet=%s status_min_secs=%s partitioning_enabled=%s output_root=%s "
        "strategy_id=%s",
        api_host,
        api_port,
        cfg.venues,
        cfg.symbols,
        cfg.interval,
        cfg.quiet,
        cfg.status_min_secs,
        cfg.partitioning_enabled,
        cfg.output_root,
        strategy_id,
    )

    # OBS: selve app-logikken ligger stadig i bot.live_connector.runner:app
    uvicorn.run(
        "bot.live_connector.runner:app",
        host=api_host,
        port=api_port,
        workers=1,
    )
