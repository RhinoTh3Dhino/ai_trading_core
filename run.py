# filepath: run.py
#!/usr/bin/env python3
"""
Universal runner for AI Trading Core.

Officiel runtime er live connector-servicen.

Eksempler:
    python run.py web
    python run.py engine-web
    python run.py script tests/test_features_pipeline.py -- --data_path data/test_data/BTCUSDT_1h_test.csv
    python run.py pytest -q tests/venues
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def _configure_logging() -> None:
    """
    Initialiser basal CLI-logging.

    Returns:
        None
    """
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_project_root() -> Path:
    """
    Skift til projektroden og eksponér den på PYTHONPATH.

    Returns:
        Projektrodens sti.
    """
    os.chdir(PROJECT_ROOT)

    pythonpath = os.environ.get("PYTHONPATH", "")
    root_str = str(PROJECT_ROOT)

    if root_str not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            os.pathsep.join([root_str, pythonpath]) if pythonpath else root_str
        )

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    logger.info("Working directory set to %s", root_str)
    return PROJECT_ROOT


def _run_subprocess(cmd: Sequence[str]) -> int:
    """
    Kør en underproces med projektets miljø.

    Args:
        cmd: Kommandoen der skal køres.

    Returns:
        Exit code fra processen.
    """
    logger.info("Executing command: %s", " ".join(cmd))

    try:
        return subprocess.call(list(cmd), env=os.environ)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to start subprocess: %s", exc)
        return 98


def _strip_remainder_leading_separator(args: list[str]) -> list[str]:
    """
    Fjern et eventuelt ledende '--' fra argparse remainder.

    Args:
        args: Argumentliste fra argparse.REMAINDER.

    Returns:
        Renset argumentliste.
    """
    if args and args[0] == "--":
        return args[1:]
    return args


def run_script(script: str, script_args: list[str]) -> int:
    """
    Kør et vilkårligt Python-script relativt til projektroden.

    Args:
        script: Sti til script relativt til projektroden.
        script_args: Ekstra argumenter til scriptet.

    Returns:
        Exit code.
    """
    _ensure_project_root()

    cleaned_args = _strip_remainder_leading_separator(script_args)
    script_path = PROJECT_ROOT / script

    if not script_path.is_file():
        logger.error("Script not found: %s", script_path)
        return 1

    cmd = [sys.executable, str(script_path), *cleaned_args]
    return _run_subprocess(cmd)


def _configure_prometheus_multiprocess(
    workers: int,
    multiproc_dir: str | None,
) -> None:
    """
    Opsæt Prometheus multiprocess-dir hvis der bruges flere workers.

    Args:
        workers: Antal workers.
        multiproc_dir: Valgfri mappe til multiprocess-metrics.

    Returns:
        None
    """
    if workers <= 1:
        return

    prom_dir = multiproc_dir or ".prom_multiproc"
    prom_path = Path(prom_dir)
    prom_path.mkdir(parents=True, exist_ok=True)

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(prom_path.resolve())
    logger.info(
        "Configured PROMETHEUS_MULTIPROC_DIR=%s",
        os.environ["PROMETHEUS_MULTIPROC_DIR"],
    )


def run_web(
    host: str,
    port: int,
    reload: bool,
    workers: int,
    multiproc_dir: str | None,
) -> int:
    """
    Start den officielle live connector-webservice.

    Args:
        host: Host for uvicorn.
        port: Port for uvicorn.
        reload: Om autoreload skal bruges.
        workers: Antal workers.
        multiproc_dir: Prometheus multiprocess-dir ved >1 workers.

    Returns:
        Exit code.
    """
    _ensure_project_root()
    _configure_prometheus_multiprocess(workers, multiproc_dir)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "bot.live_connector.runner:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    if workers > 1:
        cmd.extend(["--workers", str(workers)])

    return _run_subprocess(cmd)


def run_engine_web(
    host: str,
    port: int,
    reload: bool,
    workers: int,
    multiproc_dir: str | None,
) -> int:
    """
    Start legacy/eksperimentel engine-webservice.

    Args:
        host: Host for uvicorn.
        port: Port for uvicorn.
        reload: Om autoreload skal bruges.
        workers: Antal workers.
        multiproc_dir: Prometheus multiprocess-dir ved >1 workers.

    Returns:
        Exit code.
    """
    _ensure_project_root()
    _configure_prometheus_multiprocess(workers, multiproc_dir)

    logger.warning("engine-web is experimental and not the official runtime")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "bot.engine:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    if workers > 1:
        cmd.extend(["--workers", str(workers)])

    return _run_subprocess(cmd)


def run_pytest(py_args: list[str]) -> int:
    """
    Kør pytest med valgfri argumenter.

    Args:
        py_args: Argumenter til pytest.

    Returns:
        Exit code.
    """
    _ensure_project_root()

    cleaned_args = _strip_remainder_leading_separator(py_args)
    cmd = [sys.executable, "-m", "pytest", *cleaned_args]
    return _run_subprocess(cmd)


def build_parser() -> argparse.ArgumentParser:
    """
    Byg CLI-parseren.

    Returns:
        Konfigureret argumentparser.
    """
    parser = argparse.ArgumentParser(description="Universal runner til AI Trading Core")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_script = subparsers.add_parser("script", help="Kør et vilkårligt Python-script")
    p_script.add_argument(
        "script",
        type=str,
        help="Script-sti relativt til projektroden (fx tests/test_x.py)",
    )
    p_script.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Ekstra argumenter til scriptet",
    )

    for name, help_text in [
        ("web", "Start officiel live connector-webservice"),
        ("engine-web", "Start legacy/eksperimentel engine-webservice"),
    ]:
        p_web = subparsers.add_parser(name, help=help_text)
        p_web.add_argument("--host", type=str, default="0.0.0.0", help="Host")
        p_web.add_argument("--port", type=int, default=8000, help="Port")
        p_web.add_argument("--reload", action="store_true", help="Auto-reload i udvikling")
        p_web.add_argument("--workers", type=int, default=1, help="Antal workers")
        p_web.add_argument(
            "--multiproc-dir",
            type=str,
            default=None,
            help="Sti til Prometheus multiprocess dir",
        )

    p_pytest = subparsers.add_parser("pytest", help="Kør pytest")
    p_pytest.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Argumenter til pytest",
    )

    return parser


def main() -> int:
    """
    CLI entrypoint.

    Returns:
        Exit code for processen.
    """
    _configure_logging()

    # Bypass argparse for pytest, så flags som -q virker direkte
    if len(sys.argv) >= 2 and sys.argv[1] == "pytest":
        return run_pytest(sys.argv[2:])

    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "script":
        return run_script(args.script, args.script_args)

    if args.cmd == "web":
        return run_web(
            args.host,
            args.port,
            args.reload,
            args.workers,
            args.multiproc_dir,
        )

    if args.cmd == "engine-web":
        return run_engine_web(
            args.host,
            args.port,
            args.reload,
            args.workers,
            args.multiproc_dir,
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
