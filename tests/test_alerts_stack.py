# tests/test_alerts_stack.py
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
from datetime import datetime

# Sørg for import fra projektroden
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alerts.signal_router import SignalRouter, Decision        # noqa: E402
from alerts.alert_manager import AlertManager                  # noqa: E402
from core.state import PosState, PosSide                       # noqa: E402
import utils.telegram_utils as tg                              # noqa: E402


# ---------- test helpers ----------
class DummyClock:
    def __init__(self, t0: float | None = None):
        self.t = time.time() if t0 is None else float(t0)
    def now(self) -> float:
        return self.t
    def sleep(self, sec: float):
        self.t += float(sec)

def make_cfg():
    """
    Minimal konfig så AlertManager og SignalRouter kan køre.
    Vi bruger SimpleNamespace for dot-notation og holder samme struktur som alerts.yaml.
    """
    from types import SimpleNamespace as NS
    return NS(
        alert_manager=NS(
            dedupe_ttl_sec=90,
            dedupe_bucket_sec=60,
            cooldown_sec_global=5,
            cooldown_sec_per_symbol=3,
            batch_lowprio_every_sec=60,
            batch_max_items_preview=3,
        ),
        router=NS(
            min_qty=0.10,
            min_notional=50.0,
            min_confidence=0.55,
            urgent_confidence=0.80,
            allow_market_when_urgent=True,
            prefer_limit=True,
            lowprio=NS(enabled=True, confidence_below=0.65, types=["limit","market"]),
            notional_currency="USD",
        ),
        symbols=NS(  # eksempel override
            BTCUSDT=NS(min_qty=0.05, min_notional=25.0, min_confidence=0.50, urgent_confidence=0.80)
        ),
    )

class DummyBroker:
    """Kun med det absolut nødvendige interface for router-tests."""
    def __init__(self):
        self.positions = {
            "BTCUSDT": PosState(
                symbol="BTCUSDT",
                side=PosSide.FLAT,
                qty=0.0,
                avg_price=0.0,
                last_update_ts=datetime.utcnow(),
            )
        }

def _extract_order_type(payload) -> str | None:
    """
    Træk ordretype robust ud af payload:
    - 'order_type' (legacy)
    - top-level 'type'
    - 'data.type'
    Returnerer altid upper() hvis fundet.
    """
    if not isinstance(payload, dict):
        return None
    cand = payload.get("order_type")
    if cand is None:
        cand = payload.get("type")
    if cand is None:
        data = payload.get("data") or {}
        if isinstance(data, dict):
            cand = data.get("type")
    return str(cand).upper() if cand is not None else None


# ---------- Telegram mock ----------
class _MockResp:
    def __init__(self, ok=True, status=200, payload=None, description=""):
        self._ok = ok
        self.status_code = status
        self._payload = payload or {}
        self._desc = description
        self.text = json.dumps({"ok": ok, "description": description})
    def json(self):
        if self._payload:
            return self._payload
        return {"ok": self._ok, "description": self._desc}

def _install_requests_post_mock():
    sent = []
    real_post = tg.requests.post

    def fake_post(url, *args, **kwargs):
        # sendMessage bruger json=payload; sendPhoto/Document bruger data/files
        if "sendMessage" in url:
            payload = kwargs.get("json", {})
            sent.append(("msg", payload))
            # Simulér parse-fejl for MarkdownV2 for at teste fallback:
            pm = (payload or {}).get("parse_mode")
            if pm in ("MarkdownV2", "Markdown"):
                return _MockResp(ok=False, status=400, payload={"ok": False, "description": "Bad Request: can't parse entities"})
            # HTML fallback eller plain → OK
            return _MockResp(ok=True, status=200)
        elif "sendPhoto" in url:
            data = kwargs.get("data", {})
            sent.append(("photo", data))
            return _MockResp(ok=True, status=200)
        elif "sendDocument" in url:
            data = kwargs.get("data", {})
            sent.append(("doc", data))
            return _MockResp(ok=True, status=200)
        return _MockResp(ok=True, status=200)

    tg.requests.post = fake_post
    return sent, real_post

def _restore_requests_post(real_post):
    tg.requests.post = real_post


# ---------- tests ----------
def test_signal_router_and_alert_manager():
    cfg = make_cfg()
    clock = DummyClock(1_000_000.0)

    # AlertManager forventer en konfig-sektion og en CALLABLE clock (fx clock.now)
    am = AlertManager(cfg.alert_manager, clock.now)
    br = DummyBroker()
    router = SignalRouter(broker=br, alert_manager=am, cfg=cfg, state_store=None)

    ts = datetime(2025, 7, 1, 12, 0, 0)

    # 1) Under min_confidence → SUPPRESS
    sig_lo = {"symbol":"BTCUSDT","side":"BUY","type":"market","qty":1.0,"limit_price":None,"ts":ts,"confidence":0.4,"notional":100.0}
    d1 = router.on_signal(sig_lo)
    assert isinstance(d1, Decision) and d1.action == "SUPPRESS", f"forventede SUPPRESS, fik {d1}"

    # 2) Under min_notional → SUPPRESS
    sig_notional = {"symbol":"BTCUSDT","side":"BUY","type":"market","qty":0.2,"limit_price":None,"ts":ts,"confidence":0.9,"notional":10.0}
    d2 = router.on_signal(sig_notional)
    assert d2.action == "SUPPRESS" and "notional" in d2.reason.lower()

    # 3) Market med OK confidence men ikke urgent → skal promoveres til LIMIT når prefer_limit=True
    sig_ok = {"symbol":"BTCUSDT","side":"BUY","type":"market","qty":1.0,"limit_price":61000.0,"ts":ts,"confidence":0.70,"notional":61000.0}
    d3 = router.on_signal(sig_ok)
    assert d3.action == "NOTIFY", f"forventede NOTIFY, fik {d3}"
    typ3 = _extract_order_type(d3.payload)
    assert typ3 == "LIMIT", f"forventede LIMIT, fik {d3.payload}"

    # 4) Urgent → tillad MARKET (allow_market_when_urgent=True)
    sig_urgent = {"symbol":"BTCUSDT","side":"SELL","type":"market","qty":1.0,"limit_price":None,"ts":ts,"confidence":0.90,"notional":61000.0}
    d4 = router.on_signal(sig_urgent)
    assert d4.action == "NOTIFY", f"forventede NOTIFY, fik {d4}"
    typ4 = _extract_order_type(d4.payload)
    assert typ4 == "MARKET", f"forventede MARKET, fik {d4.payload}"

    # 5) Dedupe + cooldown
    # Første send markeres…
    assert not am.is_duplicate(sig_ok)
    am.mark_sent(sig_ok)
    # Dublet inden for TTL → True
    assert am.is_duplicate(sig_ok) is True
    # Cooldown global/symbol → in_cooldown True
    assert am.in_cooldown(sig_ok) is True
    # Fremskyd tid for at slippe cooldown (brug cfg.alert_manager.xxx værdier)
    clock.sleep(cfg.alert_manager.cooldown_sec_global + 0.1)
    assert am.in_cooldown(sig_ok) is False

def test_telegram_utils_chunk_and_fallback(tmp_path: Path | None = None):
    # Sørg for "aktiv" Telegram i test (men vi mocker netværk)
    os.environ["TELEGRAM_TOKEN"] = "TEST_TOKEN"
    os.environ["TELEGRAM_CHAT_ID"] = "12345"
    os.environ["TELEGRAM_VERBOSE"] = "0"

    sent, real_post = _install_requests_post_mock()
    try:
        # 1) Chunking: send en lang tekst > 4096 → skal blive til flere sendMessage-kald
        long_text = "A" * 5000
        tg.send_message(long_text, parse_mode=None, silent=True)
        msg_calls = [x for x in sent if x[0] == "msg"]
        assert len(msg_calls) >= 2, f"Forventede chunking til >=2 kald, fik {len(msg_calls)}"

        # 2) Fallback: MarkdownV2 → vi simulerer parse-fejl → fallback til HTML <pre>
        sent.clear()
        tg.send_message("*bold* _italic_", parse_mode="MarkdownV2", silent=True)
        # Første kald (MarkdownV2) + andet kald (fallback HTML)
        assert len(sent) >= 2, "Forventede fallback-kald nummer 2"
        # Sidste kald er fallback med HTML
        last_payload = sent[-1][1]
        assert last_payload.get("parse_mode") == "HTML" and str(last_payload.get("text","")).startswith("<pre>")
    finally:
        _restore_requests_post(real_post)


# ---------- simple runner ----------
def main():
    fails = 0
    try:
        test_signal_router_and_alert_manager()
        print("OK - SignalRouter & AlertManager")
    except AssertionError as e:
        print("FAIL - SignalRouter/AlertManager:", e); fails += 1
    try:
        test_telegram_utils_chunk_and_fallback()
        print("OK - Telegram utils (chunking + fallback)")
    except AssertionError as e:
        print("FAIL - Telegram utils:", e); fails += 1
    if fails == 0:
        print("\n✅ Alerts/Router/Telegram-tests bestået.")
    else:
        print(f"\n❌ {fails} test(s) fejlede.")
        sys.exit(1)

if __name__ == "__main__":
    main()
