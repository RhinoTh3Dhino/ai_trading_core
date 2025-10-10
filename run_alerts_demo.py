# run_alerts_demo.py
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict

# --- Sørg for import fra projektroden ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.telegram_utils as tg  # noqa: E402
from alerts.alert_manager import AlertManager  # noqa: E402
# Lokale imports
from alerts.signal_router import Decision, SignalRouter  # noqa: E402
from core.state import PosSide, PosState  # noqa: E402

# ------------------------- utils -------------------------


def _to_ns(obj: Any) -> Any:
    """Recursiv dict -> SimpleNamespace (så router/AM kan bruge dot-notation)."""
    if isinstance(obj, dict):
        return NS(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(x) for x in obj]
    return obj


def load_config(path: Path) -> Any:
    """Load YAML til SimpleNamespace. Falder tilbage til fornuftige defaults."""
    import yaml  # kræver pyyaml

    if not path.exists():
        # Fallback-defaults (matcher vores tests)
        cfg = {
            "alert_manager": {
                "dedupe_ttl_sec": 90,
                "dedupe_bucket_sec": 60,
                "cooldown_sec_global": 5,
                "cooldown_sec_per_symbol": 3,
                "batch_lowprio_every_sec": 60,
                "batch_max_items_preview": 3,
            },
            "router": {
                "min_qty": 0.10,
                "min_notional": 50.0,
                "min_confidence": 0.55,
                "urgent_confidence": 0.80,
                "allow_market_when_urgent": True,
                "prefer_limit": True,
                "lowprio": {
                    "enabled": True,
                    "confidence_below": 0.65,
                    "types": ["limit", "market"],
                },
                "price_decimals": 2,
                "qty_decimals": 8,
                "notional_currency": "USD",
            },
            "symbols": {
                "BTCUSDT": {
                    "min_qty": 0.05,
                    "min_notional": 25.0,
                    "min_confidence": 0.50,
                    "urgent_confidence": 0.80,
                },
                "ETHUSDT": {"min_qty": 0.10, "min_notional": 20.0},
            },
        }
        return _to_ns(cfg)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _to_ns(data)


def color(s: str, c: str) -> str:
    if not sys.stdout.isatty():
        return s
    pal = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
        "bold": "\033[1m",
    }
    return f"{pal.get(c,'')}{s}{pal['reset']}"


# ------------------------- demo broker -------------------------


class DemoBroker:
    """Minimal broker til router-demo (ingen eksekvering)."""

    def __init__(self):
        self.trading_halted = False
        self.positions: Dict[str, PosState] = {
            "BTCUSDT": PosState(
                symbol="BTCUSDT",
                side=PosSide.FLAT,
                qty=0.0,
                avg_price=0.0,
                last_update_ts=datetime.now(timezone.utc),
            )
        }


# ------------------------- sending -------------------------


def send_payload(payload: dict, dry_run: bool) -> None:
    """Send til Telegram hvis muligt; ellers print pænt til konsol."""
    text = payload.get("text", "")
    pmode = payload.get("parse_mode", "HTML")
    typ = payload.get("type") or (payload.get("data", {}).get("type"))
    label = f"[{typ}]" if typ else ""
    if dry_run or not tg.telegram_enabled():
        print(color("TELEGRAM (dry-run) " + label, "cyan"))
        print(text)
        print()
    else:
        tg.send_message(text, parse_mode=pmode, silent=False)


# ------------------------- scenario -------------------------


def build_demo_signals(ts: datetime) -> list[dict]:
    """Lille suite af signaler der udløser alle grene: suppress, limit, urgent market, duplicate/cooldown."""
    return [
        # 1) Under min_confidence → SUPPRESS
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "market",
            "qty": 1.0,
            "limit_price": None,
            "ts": ts,
            "confidence": 0.40,
            "notional": 100.0,
        },
        # 2) Under min_notional → SUPPRESS
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "market",
            "qty": 0.2,
            "limit_price": None,
            "ts": ts,
            "confidence": 0.90,
            "notional": 10.0,
        },
        # 3) Market + limit_price sat (router normaliserer til LIMIT)
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "market",
            "qty": 1.0,
            "limit_price": 61000.0,
            "ts": ts,
            "confidence": 0.70,
            "notional": 61000.0,
        },
        # 4) Urgent → MARKET (allow_market_when_urgent)
        {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "market",
            "qty": 1.0,
            "limit_price": None,
            "ts": ts,
            "confidence": 0.90,
            "notional": 61000.0,
        },
        # 5) Dublet af #3 → Duplicate/Cooldown
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "market",
            "qty": 1.0,
            "limit_price": 61000.0,
            "ts": ts,
            "confidence": 0.70,
            "notional": 61000.0,
        },
    ]


# ------------------------- main runner -------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Kør en demo af alerts-stakken (SignalRouter + AlertManager + Telegram)."
    )
    ap.add_argument(
        "--config",
        "-c",
        default=str(ROOT / "config" / "alerts.yaml"),
        help="Sti til config/alerts.yaml",
    )
    ap.add_argument(
        "--send",
        action="store_true",
        help="Send rigtige Telegram-beskeder (kræver TELEGRAM_TOKEN/CHAT_ID).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sekunder at sove mellem signaler (for at se cooldown i realtid).",
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    # AlertManager bruger en clock-callable → brug time.time for wall clock
    am = AlertManager(cfg.alert_manager, time.time)
    broker = DemoBroker()
    router = SignalRouter(broker=broker, alert_manager=am, cfg=cfg, state_store=None)

    ts = datetime(2025, 7, 1, 12, 0, 0)  # Demo-timestamp (UTC)
    signals = build_demo_signals(ts)

    print(color("== Alerts demo starter ==", "bold"))
    print(f"Config: {args.config}")
    print(f"Telegram enabled: {tg.telegram_enabled()}  (override med --send)")
    print()

    # Kør signalerne
    for i, sig in enumerate(signals, 1):
        print(color(f"[{i}] Signal ind:", "yellow"), sig)
        decision: Decision = router.on_signal(sig)

        if decision.action == "NOTIFY":
            print(
                color(f" -> Decision: {decision.action} ({decision.reason})", "green")
            )
            send_payload(decision.payload or {}, dry_run=not args.send)
        else:
            print(color(f" -> Decision: {decision.action} ({decision.reason})", "red"))
            # Demonstrér low-prio-batching, hvis det er pga. lav confidence og lowprio er slået til
            lpcfg = getattr(cfg.router, "lowprio", NS(enabled=False))
            c = sig.get("confidence")
            if (
                decision.reason.lower().startswith("low confidence")
                and getattr(lpcfg, "enabled", False)
                and c is not None
                and getattr(cfg.router, "min_confidence", 0.0)
                <= c
                < getattr(lpcfg, "confidence_below", 1.0)
            ):
                # læg en kompakt payload i buffer
                am.enqueue_lowprio(
                    {
                        "symbol": sig["symbol"],
                        "side": sig["side"],
                        "type": sig["type"],
                        "qty": sig.get("qty"),
                        "limit_price": sig.get("limit_price"),
                    }
                )
                print(color("    (lagt i low-prio buffer)", "cyan"))

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Forsøg at flush’e low-prio batch (force ved at snyde _last_batch_ts)
    if am.lowprio_buffer:
        am._last_batch_ts = 0.0  # force: sørg for at interval-betingelsen er sand nu
        batches = am.maybe_flush_batch()
        for b in batches:
            print(color("[LOW-PRIO BATCH]", "cyan"))
            if args.send and tg.telegram_enabled():
                tg.send_message(b["text"], parse_mode=None)
            else:
                print(b["text"])
                print()

    print(color("== Demo slut ==", "bold"))


if __name__ == "__main__":
    main()
