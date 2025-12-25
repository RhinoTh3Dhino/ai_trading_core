# bot/live_connector/strategy_loader.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Type

import yaml

from bot.strategies.flagship_trend_v1 import FlagshipTrendConfig, FlagshipTrendV1Strategy

# Registry over kendte strategier
STRATEGY_REGISTRY: Dict[str, Tuple[Type[object], Type[object]]] = {
    "flagship_trend_v1": (FlagshipTrendV1Strategy, FlagshipTrendConfig),
}


@dataclass
class StrategyBootstrap:
    id: str
    instance: object
    config: object


def load_strategy_from_env() -> StrategyBootstrap:
    """
    Loader strategi ud fra env-var LYRA_STRATEGY_ID (default: flagship_trend_v1)
    + YAML-config i config/strategies/{id}.yml
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

    params = (raw.get("params") or {}) if isinstance(raw, dict) else {}
    cfg = cfg_cls(**params)
    strategy = strategy_cls(cfg)

    return StrategyBootstrap(id=strategy_id, instance=strategy, config=cfg)
