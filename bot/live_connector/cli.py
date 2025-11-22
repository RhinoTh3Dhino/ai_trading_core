# bot/live_connector/cli.py

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from .config import LiveConfig
from .engine import run_live_connector  # tilpas til din eksisterende runner

logger = logging.getLogger(__name__)


# --------------------------- CLI parser ---------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live-connector",
        description="Kør live datafeed + streaming-features.",
    )

    # Venue / symbols / interval
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

    # Logging / quiet-mode
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
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log-niveau (overstyrer LOG_LEVEL env).",
    )

    # Output / partitionering
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


# --------------------------- util ---------------------------


def _setup_logging(log_level_arg: Optional[str]) -> None:
    """Initialiser logging baseret på CLI-flag + env LOG_LEVEL."""

    # Prioritet: CLI-flag > LOG_LEVEL env > INFO
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level_name = (log_level_arg or env_level).upper()

    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ekstra info om valgt niveau
    root = logging.getLogger()
    root.setLevel(level)
    logger.info(
        "Logging initialiseret med level=%s (env=%s, arg=%s)", level_name, env_level, log_level_arg
    )


def _debug_dump_live_env() -> None:
    """Log de vigtigste LIVE_* env-variabler for lettere fejlfinding."""
    keys = [
        "LIVE_VENUES",
        "LIVE_SYMBOLS",
        "LIVE_INTERVAL",
        "LIVE_QUIET",
        "LIVE_STATUS_MIN_SECS",
        "LIVE_PARTITIONING_ENABLED",
        "LIVE_OUTPUT_ROOT",
        "LIVE_API_HOST",
        "LIVE_API_PORT",
        "DQ_SHARED_SECRET",
    ]
    env_snapshot = {k: os.getenv(k) for k in keys}
    logger.debug("LIVE env snapshot: %s", env_snapshot)


# --------------------------- main ---------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1) Setup logging tidligt, så vi kan se konfig / fejl
    _setup_logging(args.log_level)

    _debug_dump_live_env()

    # 2) Byg LiveConfig ud fra env + CLI-args
    try:
        cfg = LiveConfig.from_env_and_args(args)
    except Exception as exc:
        logger.exception("Konfigurationsfejl i LiveConfig.from_env_and_args: %s", exc)
        return 1

    logger.info(
        "Starter live-connector med "
        "venues=%s symbols=%s interval=%s quiet=%s status_min_secs=%s "
        "partitioning_enabled=%s prometheus_port=%s output_root=%s",
        cfg.venues,
        cfg.symbols,
        cfg.interval,
        cfg.quiet,
        cfg.status_min_secs,
        cfg.partitioning_enabled,
        cfg.prometheus_port,
        cfg.output_root,
    )

    # 3) Kør selve live-runner
    try:
        run_live_connector(cfg)
    except KeyboardInterrupt:
        logger.warning("Live-connector afbrudt (KeyboardInterrupt). Lukker pænt ned.")
        return 130  # standard ctrl+c exit code
    except Exception as exc:
        logger.exception("Uventet fejl under kørsel af live-connector: %s", exc)
        return 2

    logger.info("Live-connector afsluttet uden fejl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
