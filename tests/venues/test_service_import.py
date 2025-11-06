# tests/venues/test_service_import.py

import importlib
import types


def test_import_service_module():
    m = importlib.import_module("bot.live_connector.service")
    assert isinstance(m, types.ModuleType)

    # Hvis modulet eksporterer en FastAPI-app eller builder, så rør den let
    for name in ("app", "make_app", "create_app"):
        if hasattr(m, name):
            obj = getattr(m, name)
            try:
                # callables → kald dem; ellers bare access for at ramme lazy props
                obj = obj() if callable(obj) else obj
            except Exception:
                pass
