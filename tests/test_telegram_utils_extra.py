# tests/test_telegram_utils_extra.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types
import io
import os
import pytest

import utils.telegram_utils as tg


def _enable_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "x-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")


class _Resp:
    def __init__(self, ok=True, status=200, text="OK", payload=None):
        self.ok = ok
        self.status_code = status
        self.text = text
        self._payload = payload or {"ok": ok}

    def json(self):
        return dict(self._payload)


def test_send_image_and_document_success_and_error(tmp_path, monkeypatch, capsys):
    _enable_env(monkeypatch)

    calls = {"n": 0, "last": None}

    def fake_ok(url, data=None, files=None, timeout=None, **_):
        calls["n"] += 1
        calls["last"] = ("ok", url, bool(files), data)
        return _Resp(ok=True, status=200, text="ok")

    def fake_err(url, data=None, files=None, timeout=None, **_):
        calls["n"] += 1
        calls["last"] = ("err", url, bool(files), data)
        return _Resp(ok=False, status=500, text="fail", payload={"ok": False})

    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")
    doc = tmp_path / "x.csv"
    doc.write_text("a,b\n1,2\n", encoding="utf-8")

    # success
    monkeypatch.setattr(tg, "requests", types.SimpleNamespace(post=fake_ok))
    r1 = tg.send_image(img, caption="cap")
    r2 = tg.send_document(doc, caption="doc")
    assert r1.ok and r2.ok
    assert calls["n"] == 2

    # error
    calls["n"] = 0
    monkeypatch.setattr(tg, "requests", types.SimpleNamespace(post=fake_err))
    r3 = tg.send_image(img, caption="cap")
    r4 = tg.send_document(doc, caption="doc")
    # må ikke raise
    assert calls["n"] == 2
    out = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "fejl" in out or "fail" in out or "500" in out


def test_send_trend_graph_branches(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    sent = {"msgs": [], "images": []}

    def fake_send_message(msg, **_):
        sent["msgs"].append(msg)

    def fake_send_image(path, **_):
        sent["images"].append(path)

    # a) generate_trend_graph er None
    monkeypatch.setattr(tg, "send_message", fake_send_message)
    monkeypatch.setattr(tg, "send_image", fake_send_image)
    monkeypatch.setattr(tg, "generate_trend_graph", None, raising=False)
    tg.send_trend_graph()
    assert any("ikke tilgængelig" in m.lower() for m in sent["msgs"])

    # b) generate_trend_graph returnerer et eksisterende billede
    img = tmp_path / "trend.png"
    img.write_bytes(b"\x89PNG\r\n")

    def fake_gen(**kwargs):
        return str(img)

    sent["msgs"].clear()
    sent["images"].clear()
    monkeypatch.setattr(tg, "generate_trend_graph", fake_gen, raising=False)
    tg.send_trend_graph()
    assert str(img) in sent["images"]


def test_send_live_metrics_with_alarms(monkeypatch):
    _enable_env(monkeypatch)
    # metrics der trigger alle tre alarmer
    def fake_calc(trades_df, balance_df):
        return {
            "profit_pct": -50.0,
            "win_rate": 10.0,
            "drawdown_pct": -30.0,
            "num_trades": 5,
            "profit_factor": 0.5,
            "sharpe": -1.2,
        }

    sent = {"msgs": []}

    def fake_send_message(msg, **_):
        sent["msgs"].append(msg)

    monkeypatch.setattr(tg, "calculate_live_metrics", fake_calc)
    monkeypatch.setattr(tg, "send_message", fake_send_message)

    import pandas as pd

    bal = pd.DataFrame({"balance": [1000, 900, 950]})
    tr = pd.DataFrame({"type": ["BUY", "TP", "SL"]})
    tg.send_live_metrics(
        tr,
        bal,
        symbol="X",
        timeframe="1h",
        thresholds={"drawdown": -20, "winrate": 20, "profit": -10},
    )
    # 1 summary + 3 alarmer
    assert len(sent["msgs"]) >= 4
    joined = "\n".join(sent["msgs"]).lower()
    assert "drawdown" in joined and "win-rate" in joined and "profit" in joined
