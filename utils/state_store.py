# utils/state_store.py
from __future__ import annotations

import json
import os
import threading
from collections.abc import MutableMapping as _MutableMappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, MutableMapping


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_dt_maybe(val: Any) -> datetime | None:
    """
    Forsøg at parse en værdi til datetime (UTC) hvis den ligner et timestamp.
    Understøtter ISO-8601-strenge samt epoch (int/float).
    """
    if isinstance(val, datetime):
        return _ensure_utc(val)
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        # Tolerér 'Z' suffix
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            return _ensure_utc(dt)
        except Exception:
            return None
    return None


class _DatetimeJSONEncoder(json.JSONEncoder):
    """JSON-encoder der serialiserer datetime til ISO-8601 (UTC)."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return _ensure_utc(obj).isoformat().replace("+00:00", "Z")
        return super().default(obj)


class JsonStateStore(_MutableMappingABC):
    """
    En lille, trådsikker, JSON-backet key-value store (dict-lignende) til at
    persistere router/alert-state mellem kørsler.

    Fokus:
    - Atomic skrivning (tmp+os.replace) for at undgå korruption
    - Trådsikring med RLock
    - Automatisk (de)serialisering af datetime værdier i 'router:last_sent'

    Eksempel:
        from utils.state_store import JsonStateStore
        state = JsonStateStore("outputs/alerts_state.json")
        state["router:last_hash"] = {}
        state["router:last_sent"] = {"BTCUSDT|BUY|limit": datetime.now(timezone.utc)}
        state.flush()
    """

    def __init__(self, path: str | os.PathLike, autosave: bool = False):
        self.path = Path(path)
        self.autosave = autosave
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._load()

    # ---------- MutableMapping interface ----------

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            if self.autosave:
                self._flush_locked()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]
            if self.autosave:
                self._flush_locked()

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            # iterér over et snapshot for at undgå race
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # ---------- Public helpers ----------

    def flush(self) -> None:
        """Skriv state atomisk til disk."""
        with self._lock:
            self._flush_locked()

    def clear_and_flush(self) -> None:
        """Tøm alt og skriv til disk (brug med omtanke)."""
        with self._lock:
            self._data.clear()
            self._flush_locked()

    def __repr__(self) -> str:  # debug-venlig repr
        with self._lock:
            return f"JsonStateStore(path={self.path!s}, keys={list(self._data.keys())})"

    # Context manager: auto-flush ved exit (selv ved exception)
    def __enter__(self) -> "JsonStateStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()

    # ---------- Internals ----------

    def _load(self) -> None:
        """Indlæs fra fil hvis den eksisterer, og coerce kendte felter."""
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists():
                    with self.path.open("r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            self._data.update(loaded)
                            self._postprocess_loaded()
            except json.JSONDecodeError:
                # Behold eksisterende _data (tom), men gem den korrupte fil som .corrupt
                corrupt = self.path.with_suffix(self.path.suffix + ".corrupt")
                try:
                    self.path.replace(corrupt)
                except Exception:
                    pass  # bedste forsøg
            except Exception:
                # Fail-silent: vi vil helst ikke crashe live-loop pga. state.
                pass

    def _postprocess_loaded(self) -> None:
        """
        Coerce specifikke kendte felter tilbage til korrekte typer.
        Lige nu: 'router:last_sent' værdier → datetime (UTC) hvis muligt.
        """
        last_sent = self._data.get("router:last_sent")
        if isinstance(last_sent, dict):
            for k, v in list(last_sent.items()):
                dt = _parse_dt_maybe(v)
                if dt is not None:
                    last_sent[k] = dt

    def _flush_locked(self) -> None:
        """Forventes kaldt under lås."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            # Skriv atomisk via tmp-fil og replace
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    self._data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    cls=_DatetimeJSONEncoder,
                )
            os.replace(tmp, self.path)  # atomic på de fleste platforme
        finally:
            # Best effort oprydning af tmp hvis noget gik galt
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass


# Lille convenience-funktion hvis man vil have kortere import
def load_state_store(path: str | os.PathLike, autosave: bool = False) -> JsonStateStore:
    """
    Opret og returnér en JsonStateStore.
    """
    return JsonStateStore(path=path, autosave=autosave)


if __name__ == "__main__":
    # Minimal manuel test (kører kun hvis filen køres direkte)
    s = JsonStateStore("outputs/alerts_state.json")
    s["router:last_hash"] = s.get("router:last_hash", {})
    ls = s.get("router:last_sent", {})
    ls["BTCUSDT|BUY|limit"] = datetime.now(timezone.utc)
    s["router:last_sent"] = ls
    s.flush()
    print("State gemt:", s)
