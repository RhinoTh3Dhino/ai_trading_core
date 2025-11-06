# utils/dq_wiring.py
"""
DQ-wiring (letvægts stub)
- Ingen sideeffekter ved import.
- Kalder dine metrics-helpers fra bot.live_connector.metrics hvis de findes.
- Kan valgfrit eksponere små FastAPI-ruter (/dq/*) på en eksisterende app.

Brug:
    from utils.dq_wiring import emit_violation, set_freshness, wire_fastapi_routes

    # i kode
    emit_violation("ohlcv_1h", "bounds_min", n=2)
    set_freshness("ohlcv_1h", 3.5)

    # til at montere ruter på en FastAPI-app:
    wire_fastapi_routes(app, require_secret=True)
"""

from __future__ import annotations

import os
from typing import Callable, Optional


# --- prøv at hente helpers fra dine metrics; ellers no-ops -------------------
def _get_helpers():
    try:
        from bot.live_connector.metrics import inc_dq_violation, set_dq_freshness_minutes

        def _emit(contract: str, rule: str, n: int = 1) -> None:
            for _ in range(max(1, int(n))):
                inc_dq_violation(contract, rule)

        def _fresh(dataset: str, minutes: float) -> None:
            set_dq_freshness_minutes(dataset, float(minutes))

        return _emit, _fresh
    except Exception:
        # no-op fallback (giver aldrig import-fejl)
        def _emit(contract: str, rule: str, n: int = 1) -> None:
            return None

        def _fresh(dataset: str, minutes: float) -> None:
            return None

        return _emit, _fresh


_emit_helper, _fresh_helper = _get_helpers()


# --- offentlig API -----------------------------------------------------------
def emit_violation(contract: str, rule: str, n: int = 1) -> None:
    """Inkrementér DQ-violation(s) (no-op hvis metrics ikke er tilgængelig)."""
    _emit_helper(contract, rule, n=n)


def set_freshness(dataset: str, minutes: float) -> None:
    """Sæt freshness-minutter (no-op hvis metrics ikke er tilgængelig)."""
    _fresh_helper(dataset, minutes)


# --- valgfri FastAPI-wiring (ingen afhængighed hvis ikke brugt) --------------
def wire_fastapi_routes(app, require_secret: bool = True) -> None:
    """
    Monter simple /dq/* ruter på en eksisterende FastAPI-app.
    Beskyt med header-secret hvis require_secret=True.

    Env:
        DQ_SHARED_SECRET  - hvis sat, kræves match i header 'x-dq-secret'
    """
    try:
        from fastapi import Header, HTTPException, Query
        from fastapi.responses import JSONResponse
    except Exception:
        # FastAPI ikke installeret i dette miljø; ignorer pænt.
        return

    secret_env = os.getenv("DQ_SHARED_SECRET", "").strip()

    def _auth(header_val: Optional[str]) -> bool:
        if not require_secret:
            return True
        if not secret_env:
            return True  # tillad hvis intet secret er sat i env
        return (header_val or "") == secret_env

    @app.post("/dq/freshness")
    def _dq_freshness(
        dataset: str = Query(...),
        minutes: float = Query(...),
        x_dq_secret: Optional[str] = Header(default=None),
    ):
        if not _auth(x_dq_secret):
            raise HTTPException(status_code=401, detail="Unauthorized")
        set_freshness(dataset, minutes)
        return JSONResponse({"ok": True, "dataset": dataset, "minutes": float(minutes)})

    @app.post("/dq/violation")
    def _dq_violation(
        contract: str = Query(...),
        rule: str = Query(...),
        n: int = Query(default=1, ge=1),
        x_dq_secret: Optional[str] = Header(default=None),
    ):
        if not _auth(x_dq_secret):
            raise HTTPException(status_code=401, detail="Unauthorized")
        emit_violation(contract, rule, n=n)
        return JSONResponse({"ok": True, "contract": contract, "rule": rule, "inc": int(n)})


__all__ = ["emit_violation", "set_freshness", "wire_fastapi_routes"]
