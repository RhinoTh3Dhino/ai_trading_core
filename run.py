#!/usr/bin/env python3
"""
Universal runner for AI trading bot system.

Eksempler:
  # 1) Kør et vilkårligt Python-script (som før)
  python run.py script tests/test_features_pipeline.py -- --data_path data/test_data/BTCUSDT_1h_test.csv

  # 2) Start web-appen (uvicorn bot.engine:app) på :8000
  python run.py web

  # 3) Start web-appen med 2 workers og multiprocess metrics
  python run.py web --workers 2 --multiproc-dir ./.prom_multiproc

  # 4) Kør pytest med argumenter
  python run.py pytest -q tests/test_metrics_exposition.py::test_metrics_endpoint_has_core_metrics
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_project_root() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    # Tilføj projektroden til PYTHONPATH for underprocesser
    pythonpath = os.environ.get("PYTHONPATH", "")
    if project_root not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            os.pathsep.join([project_root, pythonpath]) if pythonpath else project_root
        )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"[INFO] Working directory sat til: {project_root}")
    return project_root


def run_script(script: str, script_args: list[str]) -> int:
    project_root = _ensure_project_root()
    script_path = os.path.join(project_root, script)
    if not os.path.isfile(script_path):
        print(f"[FEJL] Scriptet findes ikke: {script_path}")
        return 1

    cmd = [sys.executable, script_path] + script_args
    print(f"[INFO] Kommando: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, env=os.environ)
        print("[OK] Script kørt færdigt uden fejl!")
        return result.returncode if result else 0
    except subprocess.CalledProcessError as e:
        print(f"[FEJL] Script fejlede med kode {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"[FEJL] Uventet fejl: {e}")
        return 99


def run_web(host: str, port: int, reload: bool, workers: int, multiproc_dir: str | None) -> int:
    _ensure_project_root()

    # Hvis multiprocess, sørg for Prometheus multiprocess-dir
    if workers and workers > 1:
        prom_dir = multiproc_dir or ".prom_multiproc"
        prom_path = Path(prom_dir)
        prom_path.mkdir(parents=True, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(prom_path.resolve())
        print(f"[INFO] PROMETHEUS_MULTIPROC_DIR sat til: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")

    # Start uvicorn mod app'en i bot/engine.py
    # VIGTIGT: Din app eksporteres som 'app' i modulet 'bot.engine'
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
    if workers and workers > 1:
        cmd.extend(["--workers", str(workers)])

    print(f"[INFO] Starter webserver: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, env=os.environ)
    except KeyboardInterrupt:
        print("\n[INFO] Stopper webserver (CTRL+C).")
        return 0
    except Exception as e:
        print(f"[FEJL] Kunne ikke starte webserver: {e}")
        return 98


def run_pytest(py_args: list[str]) -> int:
    _ensure_project_root()
    cmd = [sys.executable, "-m", "pytest"] + py_args
    print(f"[INFO] PyTest kommando: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, env=os.environ)
    except Exception as e:
        print(f"[FEJL] PyTest fejlede at starte: {e}")
        return 97


def main():
    parser = argparse.ArgumentParser(description="Universal runner til AI trading bot")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Subcommand: script (default‐adfærd fra den gamle runner)
    p_script = subparsers.add_parser("script", help="Kør et vilkårligt Python-script")
    p_script.add_argument(
        "script",
        type=str,
        help="Script-sti relativt til projektroden (fx tests/test_features_pipeline.py)",
    )
    p_script.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Ekstra argumenter til scriptet (brug -- for at skille runner fra script)",
    )

    # Subcommand: web (uvicorn bot.engine:app)
    p_web = subparsers.add_parser("web", help="Start web-appen (uvicorn bot.engine:app)")
    p_web.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p_web.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_web.add_argument("--reload", action="store_true", help="Auto-reload ved filændringer (dev)")
    p_web.add_argument("--workers", type=int, default=1, help="Antal uvicorn workers")
    p_web.add_argument(
        "--multiproc-dir",
        type=str,
        default=None,
        help="Sti til Prometheus multiprocess dir (default: ./.prom_multiproc)",
    )

    # Subcommand: pytest
    p_pytest = subparsers.add_parser("pytest", help="Kør pytest med valgfri arguments")
    p_pytest.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="Arguments videre til pytest"
    )

    args = parser.parse_args()

    if args.cmd == "script":
        rc = run_script(args.script, args.script_args)
        sys.exit(rc)
    elif args.cmd == "web":
        rc = run_web(args.host, args.port, args.reload, args.workers, args.multiproc_dir)
        sys.exit(rc)
    elif args.cmd == "pytest":
        rc = run_pytest(args.pytest_args)
        sys.exit(rc)
    else:
        print("[FEJL] Ukendt kommando.")
        sys.exit(2)


if __name__ == "__main__":
    main()
