# alerts/alert_manager.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


class AlertManager:
    """
    Håndterer:
      - Dedupe af identiske signaler (med tids-bucket)
      - Global og pr.-symbol cooldown
      - Buffering af lav-prio beskeder og periodisk batch-sammendrag

    Forventet konfig (med sikre defaults hvis de mangler). Værdier kan ligge enten på roden
    eller under cfg.alert_manager:
      - dedupe_ttl_sec: int = 60
      - dedupe_bucket_sec: int = 60
      - cooldown_sec_global: int = 10
      - cooldown_sec_per_symbol: int = 5
      - batch_lowprio_every_sec: int = 60
      - batch_max_items_preview: int = 5
    """

    def __init__(self, cfg: Any, clock):
        """
        :param cfg: objekt/dict med konfig-attributter (se docstring)
        :param clock: enten en callable der returnerer 'nu' i sekunder (time.time-lignende)
                      ELLER et objekt med en .now() metode (f.eks. DummyClock)
        """
        self.cfg = cfg
        self.clock = clock

        self.last_sent_global: float = 0.0
        self.last_sent_by_symbol: Dict[str, float] = {}
        self.dedupe_store: Dict[str, float] = {}  # key -> expiry_ts
        self.lowprio_buffer: List[Tuple[float, dict]] = []  # [(ts_sec, payload), ...]
        self._last_batch_ts: float = 0.0

    # ---------- interne helpers ----------

    def _cfg(self, name: str, default: int | float | str | None) -> Any:
        """
        Slår konfig op robust:
          - direkte på roden (cfg.name)
          - hvis ikke, så under cfg.alert_manager.name
          - understøtter både objekter (dot-notation) og dicts
        """
        # direkte på roden
        if isinstance(self.cfg, dict):
            if name in self.cfg:
                return self.cfg.get(name, default)
            am = self.cfg.get("alert_manager")
            if isinstance(am, dict) and name in am:
                return am.get(name, default)
            return default
        else:
            val = getattr(self.cfg, name, None)
            if val is not None:
                return val
            am = getattr(self.cfg, "alert_manager", None)
            if am is not None:
                if isinstance(am, dict):
                    return am.get(name, default)
                return getattr(am, name, default)
            return default

    def _now(self) -> float:
        """Monotont 'nu' i sekunder. Understøtter callable clock og clock.now()."""
        c = self.clock
        if callable(c):
            return float(c())
        if hasattr(c, "now"):
            return float(c.now())
        # fallback
        return float(time.time())

    @staticmethod
    def _to_epoch_seconds(ts: Any) -> float:
        """Understøtter både datetime og tal; falder tilbage til 0.0 ved ukendt format."""
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp()
        return 0.0

    def _bucket(self, ts: Any) -> int:
        """Læg ts i et tids-bucket (sekunder)."""
        bucket_sec = int(self._cfg("dedupe_bucket_sec", 60))
        bucket_sec = max(1, bucket_sec)
        t = int(self._to_epoch_seconds(ts))
        return (t // bucket_sec) * bucket_sec

    def _clean_dedupe_store(self) -> None:
        """Fjern udløbne dedupe-nøgler (billig best effort)."""
        now = self._now()
        to_del = [k for k, exp in self.dedupe_store.items() if exp <= now]
        for k in to_del:
            del self.dedupe_store[k]

    # ---------- offentlige metoder ----------

    def is_duplicate(self, sig: dict) -> bool:
        """
        Returnerer True hvis signalet er en dublet inden for dedupe_ttl_sec
        (nøglen bygges på symbol/side/type/limit_price + tids-bucket).
        """
        key = f'{sig.get("symbol")}|{sig.get("side")}|{sig.get("type")}|{sig.get("limit_price")}|{self._bucket(sig.get("ts"))}'
        now = self._now()
        exp = self.dedupe_store.get(key, 0.0)
        if now <= exp:
            return True
        ttl = float(self._cfg("dedupe_ttl_sec", 60))
        self.dedupe_store[key] = now + max(0.0, ttl)
        # Opportunistisk oprydning
        if len(self.dedupe_store) > 1_000:
            self._clean_dedupe_store()
        return False

    def in_cooldown(self, sig: dict) -> bool:
        """
        Returnerer True hvis enten global eller pr.-symbol cooldown ikke er udløbet.
        """
        now = self._now()
        if now - self.last_sent_global < float(self._cfg("cooldown_sec_global", 10)):
            return True
        sym = sig.get("symbol", "")
        last = self.last_sent_by_symbol.get(sym, 0.0)
        if now - last < float(self._cfg("cooldown_sec_per_symbol", 5)):
            return True
        return False

    def mark_sent(self, sig: dict) -> None:
        """Kaldes efter vi har sendt en besked (for at opdatere cooldowns)."""
        now = self._now()
        self.last_sent_global = now
        sym = sig.get("symbol", "")
        self.last_sent_by_symbol[sym] = now

    def enqueue_lowprio(self, payload: dict) -> None:
        """Læg en lav-prio payload i buffer (timestampes med clock())."""
        self.lowprio_buffer.append((self._now(), payload))

    def maybe_flush_batch(self) -> List[dict]:
        """
        Hvis tidsintervallet for batching er opnået og der ER data i bufferen,
        returneres en liste med ét sammendrag (dict). Ellers [].
        """
        interval = float(self._cfg("batch_lowprio_every_sec", 60))
        now = self._now()
        if now - self._last_batch_ts < interval:
            return []
        if not self.lowprio_buffer:
            self._last_batch_ts = now
            return []

        items = self.lowprio_buffer
        self.lowprio_buffer = []
        self._last_batch_ts = now
        return [self._summarize(items)]

    # ---------- summarizing ----------

    def _summarize(self, items: List[Tuple[float, dict]]) -> dict:
        """
        Lav et kompakt sammendrag egnet til en Telegram-besked.
        Returnerer en payload-dict – selve afsendelsen håndteres andetsteds.
        """
        count = len(items)
        t_first = min(ts for ts, _ in items)
        t_last = max(ts for ts, _ in items)

        # simple optællinger
        per_symbol: Dict[str, int] = {}
        per_side: Dict[str, int] = {}
        per_type: Dict[str, int] = {}
        preview_max = int(self._cfg("batch_max_items_preview", 5))

        preview = []
        for i, (_, p) in enumerate(items):
            sym = str(p.get("symbol", "?"))
            side = str(p.get("side", "?"))
            typ = str(p.get("type", "?"))
            per_symbol[sym] = per_symbol.get(sym, 0) + 1
            per_side[side] = per_side.get(side, 0) + 1
            per_type[typ] = per_type.get(typ, 0) + 1
            if i < preview_max:
                preview.append(
                    f"{sym} {side} {typ} qty={p.get('qty', '')} lim={p.get('limit_price', '')}"
                )

        # byg en enkel tekst
        sym_str = ", ".join(
            f"{s}×{n}"
            for s, n in sorted(per_symbol.items(), key=lambda x: (-x[1], x[0]))
        )
        side_str = ", ".join(
            f"{s}×{n}" for s, n in sorted(per_side.items(), key=lambda x: (-x[1], x[0]))
        )
        type_str = ", ".join(
            f"{t}×{n}" for t, n in sorted(per_type.items(), key=lambda x: (-x[1], x[0]))
        )

        header = f"📬 {count} lav-prio signaler (fra {count and int(t_last - t_first)}s vindue)"
        lines = [
            header,
            f"Symboler: {sym_str or '-'}",
            f"Sider: {side_str or '-'}",
            f"Typer: {type_str or '-'}",
        ]
        if preview:
            lines.append("— Eksempler —")
            lines.extend(preview)

        text = "\n".join(lines)

        return {
            "type": "lowprio_batch",
            "count": count,
            "t_first": t_first,
            "t_last": t_last,
            "per_symbol": per_symbol,
            "per_side": per_side,
            "per_type": per_type,
            "text": text,
        }
