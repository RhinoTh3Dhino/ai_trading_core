# bot/live_connector/config.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class LiveConfig:
    venues: List[str]
    symbols: List[str]
    interval: str

    quiet: bool
    status_min_secs: int

    partitioning_enabled: bool

    prometheus_port: int

    # Evt. allerede eksisterende felter i din app kan tilføjes her,
    # f.eks. output_path, log_level, etc.
    output_root: str

    @classmethod
    def from_env_and_args(cls, args: "argparse.Namespace") -> "LiveConfig":  # type: ignore[name-defined]
        """
        Prioritet:
        1) CLI-flags
        2) ENV-variabler
        3) Hardcodede defaults
        """
        venues = _parse_csv(args.venues or os.getenv("LIVE_VENUES", "binance,bybit"))
        symbols = _parse_csv(args.symbols or os.getenv("LIVE_SYMBOLS", "BTCUSDT,ETHUSDT"))
        interval = args.interval or os.getenv("LIVE_INTERVAL", "1m")

        quiet = bool(args.quiet or os.getenv("LIVE_QUIET", "1") in ("1", "true", "True"))

        status_min_secs = int(args.status_min_secs or os.getenv("LIVE_STATUS_MIN_SECS", "60"))

        partitioning_enabled = bool(
            args.partitioning
            or os.getenv("LIVE_PARTITIONING_ENABLED", "1") in ("1", "true", "True")
        )

        prometheus_port = int(args.prometheus_port or os.getenv("LIVE_PROMETHEUS_PORT", "9100"))

        output_root = args.output_root or os.getenv(
            "LIVE_OUTPUT_ROOT",
            "outputs/live",
        )

        if not venues:
            raise ValueError("LiveConfig: ingen venues angivet (CLI eller LIVE_VENUES)")

        if not symbols:
            raise ValueError("LiveConfig: ingen symbols angivet (CLI eller LIVE_SYMBOLS)")

        return cls(
            venues=venues,
            symbols=symbols,
            interval=interval,
            quiet=quiet,
            status_min_secs=status_min_secs,
            partitioning_enabled=partitioning_enabled,
            prometheus_port=prometheus_port,
            output_root=output_root,
        )
