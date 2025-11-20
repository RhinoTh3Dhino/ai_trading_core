# tests/venues/test_label_guard_smoke.py

import importlib


def test_import_label_guard_module():
    m = importlib.import_module("bot.live_connector.label_guard")
    # Hvis der findes noget offentligt API, så rør ganske let ved det
    assert hasattr(m, "__name__")
