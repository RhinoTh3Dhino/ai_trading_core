# tests/test_claude_eval_smoke.py
from __future__ import annotations

import json
import types

import pytest

import evaluation.claude_eval as ce


def test_prompt_p1_formats_without_keyerror():
    """Sikrer at p1-skabelonen kan .format(...) med kontekst, også hvis kontekst indeholder klammer."""
    ctx = "sample context with braces {like-this} and [arrays]"
    s = ce.PROMPTS["p1"].format(context=ctx)
    assert "CONTEXT" in s
    assert ctx in s


def test_extract_json_tolerates_wrapped_and_trailing_commas():
    """_extract_json skal kunne finde første {...} og fjerne trailing comma før parse."""
    text = (
        "prefix noise\n"
        '{ "edge_score": 0.4, "opportunities": [], "warnings": [], '
        '"action": "hold", "confidence": 0.5, }\n'
        "suffix"
    )
    obj = ce._extract_json(text)  # type: ignore[attr-defined]
    assert isinstance(obj, dict)
    assert ce.validate_payload(obj) is True


def test_dry_run_returns_valid_json(monkeypatch: pytest.MonkeyPatch):
    """Dry-run bruger mock: skal returnere gyldig JSON og passere validatoren."""
    # Gør konteksten stabil og med ord der trigger mock-warnings
    monkeypatch.setattr(
        ce, "build_context", lambda max_rows=100: "drawdown slippage commission"
    )
    r = ce.run_once("p1", dry_run=True, model=ce.DEFAULT_MODEL)
    assert r["valid"] is True
    obj = r["json"]
    assert ce.validate_payload(obj) is True
    # Basale nøgler
    for k in ("edge_score", "opportunities", "warnings", "action", "confidence"):
        assert k in obj


def test_api_call_parse_ok(monkeypatch: pytest.MonkeyPatch):
    """Simulér succesfuldt Claude-kald: responsestruktur med content[0].text indeholdende JSON."""
    payload_text = json.dumps(
        {
            "edge_score": 0.42,
            "opportunities": ["x"],
            "warnings": [],
            "action": "hold",
            "confidence": 0.9,
        }
    )

    class DummyResp:
        status_code = 200

        def raise_for_status(self):  # noqa: D401
            return None

        def json(self):
            return {"content": [{"type": "text", "text": payload_text}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResp()

    # Patch requests.post i det importerede modul
    monkeypatch.setattr(ce.requests, "post", fake_post)

    out = ce.call_claude("CTX", model="m", api_key="KEY", timeout=1, dry_run=False)
    assert ce.validate_payload(out) is True
    assert out["edge_score"] == 0.42
    assert out["action"] == "hold"
    assert out["confidence"] == 0.9


def test_api_exception_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch):
    """Hvis API fejler, skal vi falde tilbage til mock og stadig levere gyldig JSON."""

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(ce.requests, "post", boom)
    out = ce.call_claude("CTX", model="m", api_key="KEY", dry_run=False)
    assert ce.validate_payload(out) is True  # mock-output
    assert out["action"] in {"hold", "scale_in", "scale_out", "exit"}


def test_validate_payload_rejects_invalid():
    """Negativ tests for validatoren."""
    # Uden alle nødvendige felter
    bad1 = {"edge_score": 0.3, "opportunities": [], "warnings": [], "action": "hold"}
    assert ce.validate_payload(bad1) is False

    # Udenfor interval
    bad2 = {
        "edge_score": 1.1,
        "opportunities": [],
        "warnings": [],
        "action": "hold",
        "confidence": 0.5,
    }
    assert ce.validate_payload(bad2) is False

    # Forkert action
    bad3 = {
        "edge_score": 0.4,
        "opportunities": [],
        "warnings": [],
        "action": "buy",
        "confidence": 0.5,
    }
    assert ce.validate_payload(bad3) is False

    # Forkert typer
    bad4 = {
        "edge_score": 0.4,
        "opportunities": "x",
        "warnings": [],
        "action": "hold",
        "confidence": 0.5,
    }
    assert ce.validate_payload(bad4) is False
