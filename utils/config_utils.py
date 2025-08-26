# -*- coding: utf-8 -*-
"""
Konfigurationsvalidering uden eksterne libs.
"""
from __future__ import annotations

from .errors import InvalidConfigError

__all__ = ["validate_config"]

REQUIRED_KEYS: dict[str, type] = {
    "strategies": list,
    "data": dict,
    "trading": dict,
}


def _ensure(condition: bool, message: str) -> None:
    """Hjælpefunktion til at kaste konsistente fejl."""
    if not condition:
        raise InvalidConfigError(message)


def _non_empty_str(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def validate_config(cfg: dict) -> None:
    """
    Simpel schema-validering.

    Krav:
    - strategies: liste (>=1)
    - data: dict med "paths": {"raw", "processed"} som ikke-tomme str
    - trading: dict med "risk": {"max_position" mellem 0 og 1 inkl.}
    """
    _ensure(isinstance(cfg, dict), "Config skal være et dict.")

    # Topniveau-nøgler og typer
    for key, expected_type in REQUIRED_KEYS.items():
        _ensure(key in cfg, f"Manglende topnøgle: {key}")
        _ensure(
            isinstance(cfg[key], expected_type),
            f"Nøglen '{key}' skal være af type {expected_type.__name__}.",
        )

    # strategies
    strategies = cfg["strategies"]
    _ensure(len(strategies) > 0, "Mindst én strategi er påkrævet.")

    # data.paths
    data = cfg["data"]
    _ensure("paths" in data and isinstance(data["paths"], dict), "data.paths skal være et dict.")
    for path_key in ("raw", "processed"):
        value = data["paths"].get(path_key)
        _ensure(
            _non_empty_str(value),
            f"data.paths.{path_key} mangler eller er ugyldig streng.",
        )

    # trading.risk.max_position
    trading = cfg["trading"]
    _ensure("risk" in trading and isinstance(trading["risk"], dict), "trading.risk skal være et dict.")
    mp = trading["risk"].get("max_position", None)
    _ensure(isinstance(mp, (int, float)), "trading.risk.max_position skal være et tal (int/float).")
    _ensure(0 <= float(mp) <= 1, "trading.risk.max_position skal være mellem 0 og 1 (inkl.).")
