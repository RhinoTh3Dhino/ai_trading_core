import subprocess
import sys
import textwrap


def test_import_metrics_bootstrap_module_subprocess():
    # Kør importen i en ny Python-proces for at isolere Prometheus registry
    code = textwrap.dedent(
        """
        import importlib
        m = importlib.import_module("bot.live_connector.metrics_bootstrap")
        # Valgfrit: kør en init-funktion hvis den findes (ignorér fejl)
        for name in ("bootstrap", "bootstrap_metrics", "init_metrics"):
            if hasattr(m, name):
                try:
                    getattr(m, name)()
                except Exception:
                    pass
    """
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\\n{res.stderr}\\nstdout:\\n{res.stdout}"
