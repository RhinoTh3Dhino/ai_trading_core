# tests/test_telegram_utils.py
"""
Ekstra-robuste tests for utils/telegram_utils.py

Dækker:
- telegram_enabled(): off (mangler/env='none') og on (gyldige env)
- send_message(): noop når disabled, succes + API-fejl (500) via mock af requests.post
- log_telegram(): skriver til udskiftet LOG_PATH
- send_telegram_heartbeat(): kalder send_message én gang og printer tekst
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types
import pytest

from utils.telegram_utils import (
    telegram_enabled,
    send_message,
    log_telegram,
    send_telegram_heartbeat,
)
import utils.telegram_utils as tg  # modulreference til monkeypatch af LOG_PATH/requests


# ---------------------------------------------------------------------
# Hjælpere
# ---------------------------------------------------------------------
def _clear_env(monkeypatch):
    """Fjern alle kendte token/chat env-keys for deterministisk adfærd."""
    for k in [
        "TELEGRAM_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]:
        monkeypatch.delenv(k, raising=False)


def _set_env(monkeypatch, token: str | None, chat_id: str | None):
    """Sæt begge varianter af env-navne, så implementeringen uanset valg bliver dækket."""
    if token is None:
        monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_TOKEN", token)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)

    if chat_id is None:
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_CHAT_ID", chat_id)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_telegram_enabled_false(monkeypatch):
    _clear_env(monkeypatch)
    # Helt uden env
    assert not bool(telegram_enabled()), "Forventede disabled når env ikke er sat"

    # 'none' som sentinel (flere codebases bruger det som 'off')
    _set_env(monkeypatch, token="none", chat_id="none")
    assert not bool(telegram_enabled()), "Forventede disabled for 'none'/'none'"


def test_telegram_enabled_true(monkeypatch):
    _set_env(monkeypatch, token="valid_token", chat_id="12345")
    assert bool(telegram_enabled()) is True, "Burde være enabled med gyldige env"


def test_send_message_noop_when_disabled(monkeypatch, capsys):
    _set_env(monkeypatch, token="none", chat_id="none")
    # Ingen requests – blot en venlig print/log i testmode
    res = send_message("Testbesked fra test")
    printed = (capsys.readouterr().out + capsys.readouterr().err).lower()
    # Returværdi kan være None/False; vi kræver blot ingen exception og noget output
    assert "testmode" in printed or "disabled" in printed or "no-op" in printed
    assert "testbesked" in printed
    assert res is None or res is False


def test_send_message_success_and_api_error(monkeypatch, capsys):
    # Aktiveret med gyldige env
    _set_env(monkeypatch, token="x-token", chat_id="12345")

    calls = {"n": 0}

    class RespOK:
        status_code = 200

        def json(self):
            return {"ok": True}

    class RespErr:
        status_code = 500

        def json(self):
            return {"ok": False, "error": "oops"}

    def fake_post_ok(url, json=None, timeout=None):
        calls["n"] += 1
        return RespOK()

    def fake_post_err(url, json=None, timeout=None):
        calls["n"] += 1
        return RespErr()

    # Succes-case
    monkeypatch.setattr(tg, "requests", types.SimpleNamespace(post=fake_post_ok))
    r1 = send_message("hej verden")
    assert calls["n"] == 1
    # Returværdi er implementeringsafhængig; blot ingen exception:
    assert r1 in (True, None, {"ok": True}) or r1 is None

    # Fejl-case (500) – må ikke crashe
    calls["n"] = 0
    monkeypatch.setattr(tg, "requests", types.SimpleNamespace(post=fake_post_err))
    try:
        r2 = send_message("fail please")
        assert calls["n"] == 1
    except Exception:
        pytest.fail("send_message bør ikke raise ved API-fejl (status 500)")
    msg = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "500" in msg or "error" in msg or "fail" in msg or "warning" in msg


def test_log_telegram_creates_log(tmp_path):
    # Peg LOG_PATH til tmp og verificér indhold
    log_file = tmp_path / "telegram_test_log.txt"
    old = getattr(tg, "LOG_PATH", str(log_file))
    try:
        tg.LOG_PATH = str(log_file)
        msg = "Log besked fra test"
        log_telegram(msg)
        assert log_file.exists(), "Log-fil blev ikke oprettet"
        content = log_file.read_text(encoding="utf-8")
        assert msg in content, "Log-besked mangler i logfil"
    finally:
        tg.LOG_PATH = old


def test_send_telegram_heartbeat(monkeypatch, capsys):
    # Heartbeat når disabled → skal printe en besked i testmode/no-op
    _set_env(monkeypatch, token="none", chat_id="none")
    send_telegram_heartbeat()
    txt = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "testmode" in txt or "disabled" in txt or "no-op" in txt
    assert "heartbeat" in txt or "kører stadig" in txt or "alive" in txt


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
