#!/usr/bin/env python3
"""
Universal runner for AI trading bot system.
Eksempel:
    python run.py tests/test_features_pipeline.py -- --data_path data/test_data/BTCUSDT_1h_test.csv
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Universal runner til AI trading bot")
    parser.add_argument(
        "script",
        type=str,
        help="Script-sti relativt til projektroden (fx tests/test_features_pipeline.py)",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Ekstra argumenter til scriptet (brug -- for at skille runner fra script)",
    )
    args = parser.parse_args()

    # Find og sæt working directory til projektroden
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    print(f"[INFO] Working directory sat til: {project_root}")

    # Tilføj projektroden til sys.path (for barneprocesser også)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if project_root not in pythonpath.split(os.pathsep):
        pythonpath = (
            os.pathsep.join([project_root, pythonpath]) if pythonpath else project_root
        )
        os.environ["PYTHONPATH"] = pythonpath
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))

    # Valider script path
    script_path = os.path.join(project_root, args.script)
    if not os.path.isfile(script_path):
        print(f"[FEJL] Scriptet findes ikke: {script_path}")
        sys.exit(1)

    # Byg kommandoen
    cmd = [sys.executable, script_path] + args.script_args
    print(f"[INFO] Kommando: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, env=os.environ)
        print("[OK] Script kørt færdigt uden fejl!")
    except subprocess.CalledProcessError as e:
        print(f"[FEJL] Script fejlede med kode {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[FEJL] Uventet fejl: {e}")
        sys.exit(99)


if __name__ == "__main__":
    main()
