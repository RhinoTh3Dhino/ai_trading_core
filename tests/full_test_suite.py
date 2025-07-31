# tests/full_test_suite.py
"""
Kører automatisk alle tests i tests/-mappen via run.py-runner
Sikrer at alle test-scripts loader korrekt uden sys.path-hacks.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import os
import subprocess
import sys
import glob

RUNNER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run.py"))
TESTS_DIR = os.path.abspath(os.path.dirname(__file__))


def run_test_script(script_path):
    rel_path = os.path.relpath(script_path, start=os.path.dirname(RUNNER))
    cmd = [sys.executable, RUNNER, rel_path]
    print(f"\n[TEST] Kører: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[OK] Test bestået: {rel_path}")
    else:
        print(f"[FEJL] Test fejlede: {rel_path} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    # Find alle test_*.py scripts i tests/
    test_scripts = sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py")))
    if not test_scripts:
        print("[FEJL] Ingen test-scripts fundet i tests/-mappen!")
        sys.exit(1)
    print(f"[INFO] Finder {len(test_scripts)} test-scripts i tests/:")
    for script in test_scripts:
        print("  -", os.path.basename(script))

    # Kør alle tests
    for script in test_scripts:
        run_test_script(script)

    print("\n🎉 Alle tests bestået!")


if __name__ == "__main__":
    main()
