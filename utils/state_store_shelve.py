# utils/state_store_shelve.py
from __future__ import annotations

import shelve
from pathlib import Path
from typing import Any, MutableMapping, Iterator


class ShelveStore(MutableMapping[str, Any]):
    """
    Minimal, fil-baseret key/value-store (dict-lignende) til vedvarende router-state.
    - Auto-commit på hver write (set/del)
    - Kan bruges som context manager: with ShelveStore(path) as store: ...
    - Trådsikkerhed er ikke garanteret (samme som 'shelve'); ét proces-skriver er fint.
    """

    def __init__(self, path: str | Path):
        self._path = str(path)
        # writeback=False for at undgå uventede memory-caches
        self._shelf = shelve.open(self._path, flag="c", writeback=False)

    # --- MutableMapping interface ---
    def __getitem__(self, key: str) -> Any:
        return self._shelf[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._shelf[key] = value
        self._shelf.sync()

    def __delitem__(self, key: str) -> None:
        del self._shelf[key]
        self._shelf.sync()

    def __iter__(self) -> Iterator[str]:
        return iter(self._shelf)

    def __len__(self) -> int:
        return len(self._shelf)

    # --- convenience ---
    def get(self, key: str, default: Any = None) -> Any:
        return self._shelf.get(key, default)

    def close(self) -> None:
        try:
            self._shelf.sync()
        finally:
            self._shelf.close()

    # --- context manager ---
    def __enter__(self) -> "ShelveStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
