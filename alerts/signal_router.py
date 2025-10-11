# alerts/signal_router.py
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


@dataclass
class Decision:
    action: str  # "SUPPRESS" | "NOTIFY"
    reason: str
    payload: Optional[Dict[str, Any]] = None  # det der skal i telegram-beskeden


# -------------------- cfg helpers --------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Hent config-værdi fra dict eller objekt (fx SimpleNamespace)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first_of(cfg_like: Any, keys: list[str], default: Any) -> Any:
    """Find første nøgle der findes i cfg_like (dict/namespace)."""
    for k in keys:
        v = _get(cfg_like, k, None)
        if v is not None:
            return v
    return default


def _resolve_cfg(cfg_root: Any, keys: list[str], default: Any) -> Any:
    """
    Søg efter en config-værdi i flere mulige namespaces:
    - roden (cfg)
    - cfg.router, cfg.alerts, cfg.signal, cfg.filters
    - cfg.settings, cfg.params
    Returnér første match eller default.
    """
    contexts = [
        cfg_root,
        _get(cfg_root, "router"),
        _get(cfg_root, "alerts"),
        _get(cfg_root, "signal"),
        _get(cfg_root, "filters"),
        _get(cfg_root, "settings"),
        _get(cfg_root, "params"),
    ]
    for ctx in contexts:
        if ctx is None:
            continue
        v = _first_of(ctx, keys, None)
        if v is not None:
            return v
    return default


# -------------------- router --------------------


class SignalRouter:
    """
    Enkel router der tager et råt 'sig'-dict og afgør om vi skal NOTIFY eller SUPPRESS.

    Forventet input:
        sig = {
            "symbol": "BTCUSDT",
            "side": "BUY" / "SELL",
            "type": "market" / "limit",
            "qty": 0.3,
            "limit_price": 61000.0 | None,
            "mkt_price": 60950.0 | None,   # valgfrit; brugt til notional for market
            "ts": datetime | None,         # bar/udløser-tid (naiv eller aware)
            "confidence": 0.74 | None,
            "notional": float | None,
        }

    Konfig (dict eller SimpleNamespace), med aliaser og understøttelse af sub-namespaces
    som .filters, .router, .alerts, .signal:
      - cooldown_sec | cooldown
      - min_confidence | min_conf | confidence_min
      - min_qty | qty_min
      - min_notional | min_notional_usd | notional_min
      - price_decimals | px_decimals
      - qty_decimals
    """

    def __init__(
        self,
        broker,
        alert_manager,
        cfg: Optional[Dict[str, Any]] = None,
        state_store: Optional[Dict[str, Any]] = None,
    ):
        self.broker = broker
        self.alert_manager = alert_manager
        self.cfg = cfg or {}
        self.state = state_store if state_store is not None else {}

        # --- konfig med aliaser + namespaces ---
        self.cooldown = int(_resolve_cfg(self.cfg, ["cooldown_sec", "cooldown"], 60))
        self.min_conf = float(
            _resolve_cfg(self.cfg, ["min_confidence", "min_conf", "confidence_min"], 0.0)
        )
        self.min_qty = float(_resolve_cfg(self.cfg, ["min_qty", "qty_min"], 0.0))
        self.min_notional = float(
            _resolve_cfg(self.cfg, ["min_notional", "min_notional_usd", "notional_min"], 0.0)
        )
        self.px_dec = int(_resolve_cfg(self.cfg, ["price_decimals", "px_decimals"], 2))
        self.qty_dec = int(_resolve_cfg(self.cfg, ["qty_decimals"], 8))

        # state-felter (lokal fallback for cooldown/dedup)
        self._last_sent: Dict[str, datetime] = self.state.get("router:last_sent") or {}
        self._last_hash: Dict[str, str] = self.state.get("router:last_hash") or {}

    # ------------------------- public API -------------------------

    def on_signal(self, sig: Dict[str, Any]) -> Decision:
        # 1) Normaliser & valider
        norm, err = self._normalize(sig)
        if err:
            return Decision("SUPPRESS", err, None)

        # 2) Global stop: trading halted?
        if getattr(self.broker, "trading_halted", False):
            return Decision("SUPPRESS", "Trading halted by risk control", None)

        # 3) Filtre (confidence/qty/notional)
        reason = self._screen(norm)
        if reason:
            return Decision("SUPPRESS", reason, None)

        # 4) Dedupe — brug AlertManager hvis tilgængelig (men IKKE dens cooldown her)
        if self.alert_manager is not None and hasattr(self.alert_manager, "is_duplicate"):
            try:
                if self.alert_manager.is_duplicate(norm):
                    return Decision("SUPPRESS", "Duplicate", None)
            except Exception:
                # fortsæt med lokal fallback hvis AlertManager fejler
                pass

        # 5) Lokal cooldown & dedup
        key = self._key(norm)
        now = self._now_utc()
        if self._is_in_cooldown_local(key, now):
            return Decision("SUPPRESS", "Cooldown", None)
        sig_hash = self._hash_sig(norm)
        last_hash = self._last_hash.get(key)
        if last_hash == sig_hash:
            return Decision("SUPPRESS", "Duplicate", None)

        # 6) Build payload (tekst til Telegram i HTML)
        payload = self._build_payload(norm)

        # 7) Commit state (lokal)
        self._last_sent[key] = now
        self._last_hash[key] = sig_hash
        self._persist_state()

        # 8) Markér sendt i AlertManager (så dens globale/symbol-cooldowns virker efterfølgende)
        if self.alert_manager is not None and hasattr(self.alert_manager, "mark_sent"):
            try:
                self.alert_manager.mark_sent(norm)
            except Exception:
                pass

        return Decision("NOTIFY", "ok", payload)

    # ------------------------- helpers -------------------------

    def _normalize(self, sig: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str]]:
        out: Dict[str, Any] = dict(sig or {})

        # priser først (kan ændre type)
        lim_px = out.get("limit_price")
        out["limit_price"] = float(lim_px) if lim_px is not None else None
        mkt_px = out.get("mkt_price")
        out["mkt_price"] = float(mkt_px) if mkt_px is not None else None

        # symbol/side/type
        sym = (out.get("symbol") or "").strip()
        side = (out.get("side") or "").upper().strip()
        typ_raw = (out.get("type") or "").lower().strip()

        if not sym:
            return {}, "Missing symbol"
        if side not in {"BUY", "SELL"}:
            return {}, "Invalid side"

        # Hvis der er limit_price, så skal det behandles som LIMIT
        if out["limit_price"] is not None:
            typ = "limit"
        else:
            typ = "market" if typ_raw not in {"market", "limit"} else typ_raw

        # qty
        try:
            qty = float(out.get("qty", 0.0))
        except Exception:
            return {}, "Invalid qty"
        out["qty"] = qty

        # ts → aware UTC
        ts = out.get("ts")
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)  # antag UTC hvis naiv
            else:
                ts = ts.astimezone(timezone.utc)
        else:
            ts = self._now_utc()
        out["ts"] = ts

        # confidence
        conf = out.get("confidence")
        out["confidence"] = float(conf) if conf is not None else None

        # notional
        notional = out.get("notional")
        if notional is None:
            px = out["limit_price"] if typ == "limit" else (out["mkt_price"] or out["limit_price"])
            if px is not None:
                notional = abs(qty) * float(px)

        out["notional"] = float(notional) if notional is not None else None
        out["symbol"] = sym
        out["side"] = side
        out["type"] = typ
        return out, None

    def _screen(self, s: Dict[str, Any]) -> Optional[str]:
        # confidence
        c = s.get("confidence")
        if c is not None and c < self.min_conf:
            return f"Low confidence ({c:.2f} < {self.min_conf:.2f})"
        # qty
        if self.min_qty > 0.0 and abs(float(s["qty"])) < self.min_qty:
            return f"Qty below min ({s['qty']} < {self.min_qty})"
        # notional
        n = s.get("notional")
        if self.min_notional > 0.0 and (n is None or n < self.min_notional):
            return f"Notional below min ({n} < {self.min_notional})"
        return None

    def _is_in_cooldown_local(self, key: str, now: datetime) -> bool:
        if self.cooldown <= 0:
            return False
        last = self._last_sent.get(key)
        if not last:
            return False
        return (now - last) < timedelta(seconds=self.cooldown)

    def _hash_sig(self, s: Dict[str, Any]) -> str:
        # grov hash af de felter der typisk definerer et signal
        lp = s.get("limit_price")
        mp = s.get("mkt_price")
        return f"{s['symbol']}|{s['side']}|{s['type']}|{round(float(s['qty']), self.qty_dec)}|{lp}|{mp}|{round((s.get('confidence') or 0.0), 4)}"

    def _key(self, s: Dict[str, Any]) -> str:
        # cooldown per symbol/side/type
        return f"{s['symbol']}|{s['side']}|{s['type']}"

    def _persist_state(self) -> None:
        # hvis vi fik et eksternt store, læg det tilbage
        self.state["router:last_sent"] = self._last_sent
        self.state["router:last_hash"] = self._last_hash

    # -------- payload / formatting --------

    def _build_payload(self, s: Dict[str, Any]) -> Dict[str, Any]:
        sym = html.escape(s["symbol"])
        side = html.escape(s["side"])
        typ_code = s["type"]  # 'limit' / 'market' (lowercase)
        typ_disp = typ_code.upper()  # til visning OG evt. bagud-kompatibilitet
        qty = round(float(s["qty"]), self.qty_dec)
        lim_px = s.get("limit_price")
        mkt_px = s.get("mkt_price")
        conf = s.get("confidence")
        notional = s.get("notional")
        ts_utc = (
            s["ts"].astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )

        px_part = ""
        if typ_code == "limit" and lim_px is not None:
            px_part = f" @ {round(float(lim_px), self.px_dec)}"
        elif typ_code == "market" and mkt_px is not None:
            px_part = f" @ ~{round(float(mkt_px), self.px_dec)}"

        lines = [
            f"📣 <b>Trade signal</b>",
            f"• Symbol: <b>{sym}</b>",
            f"• Retning: <b>{side}</b>",
            f"• Type: <b>{typ_disp}{px_part}</b>",
            f"• Qty: <b>{qty}</b>",
        ]
        if notional is not None:
            lines.append(f"• Notional: <b>{round(float(notional), self.px_dec)}</b>")
        if conf is not None:
            lines.append(f"• Confidence: <b>{conf:.2f}</b>")
        lines.append(f"• TS (UTC): <code>{ts_utc}</code>")

        text = "\n".join(lines)
        return {
            "text": text,
            "parse_mode": "HTML",
            "order_type": typ_code,  # <-- testen tjekker denne nøgle ("limit"/"market")
            "type": typ_disp,  # bagud-kompatibelt top-niveau (UPPERCASE)
            # rå data (bevar type i lowercase her for machine use)
            "data": {
                "symbol": s["symbol"],
                "side": s["side"],
                "type": typ_code,
                "qty": qty,
                "limit_price": s.get("limit_price"),
                "mkt_price": s.get("mkt_price"),
                "notional": notional,
                "confidence": conf,
                "ts_utc": ts_utc,
            },
        }

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)
