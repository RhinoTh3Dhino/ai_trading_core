# tests/test_telegram_utils.py

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import os
import pytest
from utils.telegram_utils import (
    telegram_enabled,
    send_message,
    log_telegram,
    send_telegram_heartbeat,
    LOG_PATH,
)


def test_telegram_enabled_false(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "none")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "none")
    assert not telegram_enabled(), "telegram_enabled burde returnere False når variabler mangler"
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy_id")
    assert not telegram_enabled(), "telegram_enabled burde returnere False for dummy_token/dummy_id"


def test_telegram_enabled_true(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "valid_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "valid_id")
    assert telegram_enabled(), "telegram_enabled burde returnere True med valide værdier"


def test_send_message_noop(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_TOKEN", "none")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "none")
    result = send_message("Testbesked fra test")
    assert result is None, "send_message() skal returnere None i testmode"
    captured = capsys.readouterr()
    assert "[TESTMODE]" in captured.out, "Forventer [TESTMODE]-output når Telegram er deaktiveret"
    assert "Testbesked fra test" in captured.out, "Testbesked skal fremgå i output"


def test_log_telegram_creates_log(tmp_path):
    log_file = tmp_path / "telegram_test_log.txt"
    msg = "Log besked fra test"
    orig_log_path = LOG_PATH
    try:
        from utils import telegram_utils
        telegram_utils.LOG_PATH = str(log_file)
        log_telegram(msg)
        assert log_file.exists(), "Log-fil blev ikke oprettet"
        content = log_file.read_text(encoding="utf-8")
        assert msg in content, "Log besked findes ikke i log-filen"
    finally:
        telegram_utils.LOG_PATH = orig_log_path


def test_send_telegram_heartbeat(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_TOKEN", "none")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "none")
    send_telegram_heartbeat()
    captured = capsys.readouterr()
    assert "[testmode]" in captured.out.lower(), "Testmode markering mangler"
    assert "botten kører stadig" in captured.out.lower(), "Heartbeat besked mangler i output"


if __name__ == "__main__":
    pytest.main([__file__])