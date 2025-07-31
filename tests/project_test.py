"""
Samlet projekt-test for AI trading bot.
Kører alle relevante test-scripts via run.py og stopper ved første fejl.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
from pathlib import Path
import subprocess
import sys
import os

PROJECT_ROOT = Path(__file__).parent.parent  # AUTO-FIXED PATHLIB


def run_script(script_path, extra_args=""):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "run.py"),
        script_path,
    ] + extra_args.split()
    print(f"\n➡️ Starter test: {script_path}")
    print(f"[INFO] Kommando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {script_path} kørt færdigt uden fejl!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ FEJL ved {script_path}, stop: Exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    print("=== Kører fuld projekt-test ===\n")

    # ✅ Test 1: Features Pipeline Test
    run_script("tests/test_features_pipeline.py", "--symbol BTC --timeframe 1h")

    # ✅ Test 2: Model Træning via trainers
    run_script(
        "trainers/train_lightgbm.py",
        "--data data/test_data/BTCUSDT_1h_test.csv --n_estimators 5",
    )

    # ✅ Test 3: Backtest Test
    run_script("tests/test_backtest.py", "--symbol BTC --days 30")

    # ✅ Test 4: Ensemble Predict Test
    run_script("tests/test_ensemble_predict.py")

    # ✅ Test 5: Walkforward Test
    run_script("tests/test_walkforward.py")

    print("\n🎉 ✅ Alle tests kørt færdigt uden fejl – projekt OK!")
