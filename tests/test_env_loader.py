import os
from config.env_loader import load_config

def test_load_config_defaults(monkeypatch):
    for k in list(os.environ.keys()):
        if k.startswith(("ALLOW_ALERTS","ALERT_","TELEGRAM_","ENGINE_","LOG_DIR")):
            os.environ.pop(k, None)
    cfg = load_config()
    assert cfg.alerts.dd_pct == 10.0
    assert cfg.telegram.verbosity in {"none","bar","trade","alerts"}
