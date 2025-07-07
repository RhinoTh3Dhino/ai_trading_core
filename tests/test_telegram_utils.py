import os
import pytest
from utils import telegram_utils

def test_telegram_enabled_false(monkeypatch):
    # Tving Telegram til at være deaktiveret
    monkeypatch.setenv("TELEGRAM_TOKEN", "none")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "none")
    assert not telegram_utils.telegram_enabled(), "telegram_enabled burde returnere False når variabler mangler"
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy_id")
    assert not telegram_utils.telegram_enabled(), "telegram_enabled burde returnere False for dummy_token/dummy_id"

def test_telegram_enabled_true(monkeypatch):
    # Brug ikke-reserverede værdier
    monkeypatch.setenv("TELEGRAM_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_id")
    assert telegram_utils.telegram_enabled(), "telegram_enabled burde returnere True for ikke-tomme og ikke-reserverede værdier"

def test_send_message_noop(monkeypatch, capsys):
    # Sikrer at send_message ikke fejler når Telegram er inaktiv
    monkeypatch.setenv("TELEGRAM_TOKEN", "none")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "none")
    result = telegram_utils.send_message("Testbesked", silent=False)
    assert result is None
    captured = capsys.readouterr()
    assert "[CI/test]" in captured.out or "[CI/test]" in captured.err

def test_log_telegram_creates_log(tmp_path):
    # Sikrer at log_telegram faktisk skriver til fil
    log_path = tmp_path / "telegram_log.txt"
    msg = "Logtest"
    # Midlertidig override af global log_path
    orig_log_path = telegram_utils.LOG_PATH
    telegram_utils.LOG_PATH = str(log_path)
    telegram_utils.log_telegram(msg)
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert msg in content
    telegram_utils.LOG_PATH = orig_log_path  # Restore

def test_send_telegram_heartbeat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "none")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "none")
    telegram_utils.send_telegram_heartbeat()
