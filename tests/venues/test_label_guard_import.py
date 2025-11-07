# tests/venues/test_label_guard_import.py

import importlib
import types


def test_import_label_guard_module():
    m = importlib.import_module("bot.live_connector.label_guard")
    assert isinstance(m, types.ModuleType)

    # Prøv forsigtigt at kalde en evt. helper, hvis den findes
    for name in ("sanitize", "sanitize_labels", "guard", "guard_labels"):
        if hasattr(m, name):
            fn = getattr(m, name)
            try:
                _ = fn({"ok": "yes"})  # signatur-agnostisk; ignorer fejl
            except Exception:
                pass
