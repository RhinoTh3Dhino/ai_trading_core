# bot/live_connector/cli.py

from __future__ import annotations

import argparse
import logging
import sys

from .config import LiveConfig
from .engine import run_live_connector  # tilpas til din eksisterende runner

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live-connector",
        description="Kør live datafeed + streaming-features.",
    )

    parser.add_argument(
        "--venues",
        type=str,
        default=None,
        help="Kommasepareret liste af venues (overstyrer LIVE_VENUES).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Kommasepareret liste af symbols (overstyrer LIVE_SYMBOLS).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Bar-interval, fx '1m' (overstyrer LIVE_INTERVAL).",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="QUIET mode: reduceret logging, kun status-linje hver N sekunder.",
    )
    parser.add_argument(
        "--status-min-secs",
        type=int,
        default=None,
        help="Min. antal sekunder mellem status-linjer (overstyrer LIVE_STATUS_MIN_SECS).",
    )

    parser.add_argument(
        "--no-partitioning",
        dest="partitioning",
        action="store_false",
        help="Slå partitionering fra (ellers styres af LIVE_PARTITIONING_ENABLED).",
    )
    parser.set_defaults(partitioning=None)

    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=None,
        help="Port til /metrics (overstyrer LIVE_PROMETHEUS_PORT).",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Rodmappe for outputs (overstyrer LIVE_OUTPUT_ROOT).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = LiveConfig.from_env_and_args(args)
    except Exception as exc:
        logger.error("Konfigurationsfejl: %s", exc)
        return 1

    logger.info(
        "Starter live-connector med venues=%s symbols=%s interval=%s quiet=%s "
        "status_min_secs=%s partitioning_enabled=%s prometheus_port=%s output_root=%s",
        cfg.venues,
        cfg.symbols,
        cfg.interval,
        cfg.quiet,
        cfg.status_min_secs,
        cfg.partitioning_enabled,
        cfg.prometheus_port,
        cfg.output_root,
    )

    # Delegér til din eksisterende live-runner
    run_live_connector(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
