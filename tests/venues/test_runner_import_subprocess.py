# tests/venues/test_runner_import_subprocess.py

# kører som separat Python-proces, så prometheus-registry ikke clashes
import json
import os
import subprocess
import sys
import textwrap


def test_import_runner_module_subprocess():
    code = textwrap.dedent(
        """
        import importlib, json
        m = importlib.import_module("bot.live_connector.runner")
        # bare bekræft at app findes – ingen serverstart
        ok = hasattr(m, "app")
        print(json.dumps({"ok": bool(ok)}))
    """
    )
    # vigtig: pytest-cov opsamler subprocess coverage automatisk
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    out = json.loads(p.stdout.strip())
    assert out["ok"] is True
