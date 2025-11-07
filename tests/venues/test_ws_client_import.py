# tests/venues/test_ws_client_import.py

import importlib
import types


def test_import_ws_client_module():
    m = importlib.import_module("bot.live_connector.ws_client")
    assert isinstance(m, types.ModuleType)

    # Hvis der findes en klientklasse, instantiér "tomt" hvis muligt
    for name in ("WSClient", "WebSocketClient", "Client"):
        if hasattr(m, name):
            cls = getattr(m, name)
            try:
                _ = cls()  # hvis konstruktør kræver parametre, så ignorer vi fejl
            except Exception:
                pass
