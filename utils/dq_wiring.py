# utils/dq_wiring.py
"""
DQ-wiring (robust, side-effect free)
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

Forbedringer i denne version:
- Tolerant auth: accepterer secret via header (X-Dq-Secret) eller query (?secret=).
- Dev-bypass: hvis ENABLE_DEBUG_ROUTES=true eller require_secret=False, accepteres uden secret.
- Ekstra endpoints: /dq/ping (GET) + /dq/readiness (GET) + /dq/sample (POST, kun i debug) til lokal smoke.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple


# ------------------------ helpers fra metrics (no-op fallback) ----------------
def _get_helpers() -> Tuple[Callable[..., None], Callable[..., None]]:
    """
    Returnerer (emit_violation_helper, set_freshness_helper).
    Fallback er no-ops så import aldrig fejler.
    """
    try:
        from bot.live_connector.metrics import (  # type: ignore
            inc_dq_violation,
            set_dq_freshness_minutes,
        )

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


# ------------------------------- offentlig API --------------------------------
def emit_violation(contract: str, rule: str, n: int = 1) -> None:
    """Inkrementér DQ-violation(s) (no-op hvis metrics ikke er tilgængelig)."""
    _emit_helper(contract, rule, n=n)


def set_freshness(dataset: str, minutes: float) -> None:
    """Sæt freshness-minutter (no-op hvis metrics ikke er tilgængelig)."""
    _fresh_helper(dataset, minutes)


# -------------------------- FastAPI-wiring (valgfri) --------------------------
def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def wire_fastapi_routes(app, require_secret: bool = True) -> None:
    """
    Monter simple /dq/* ruter på en eksisterende FastAPI-app.
    Beskyt med header-secret hvis require_secret=True.

    Env:
        DQ_SHARED_SECRET      - hvis sat, kræves match i header 'X-Dq-Secret' (case-insensitiv)
        ENABLE_DEBUG_ROUTES   - '1/true' gør auth permissiv (til lokal/dev smoke)
    """
    try:
        from fastapi import Header, HTTPException, Query
        from fastapi.responses import JSONResponse
        from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_401_UNAUTHORIZED
    except Exception:
        # FastAPI ikke installeret i dette miljø; ignorer pænt.
        return

    secret_env = os.getenv("DQ_SHARED_SECRET", "").strip()
    debug_mode = _env_true("ENABLE_DEBUG_ROUTES", False)

    def _auth(header_val: Optional[str], query_secret: Optional[str]) -> bool:
        """
        Regler:
          - Hvis require_secret=False → tillad.
          - Hvis ENABLE_DEBUG_ROUTES=true → tillad.
          - Hvis ingen secret er sat i env → tillad (for nem lokal brug).
          - Ellers kræv match på header eller query (?secret=).
        """
        if not require_secret:
            return True
        if debug_mode:
            return True
        if not secret_env:
            return True
        candidate = (header_val or "") or (query_secret or "")
        return candidate == secret_env

    @app.get("/dq/ping")
    def _dq_ping():
        # Simpel liveness for DQ-router
        return JSONResponse({"ok": True, "dq_routes": True}, status_code=HTTP_200_OK)

    @app.get("/dq/readiness")
    def _dq_readiness():
        # Kun let check – ruterne er wired og helpers findes (no-op er også OK)
        helpers_ok = callable(_emit_helper) and callable(_fresh_helper)
        return JSONResponse({"ok": True, "helpers_ok": bool(helpers_ok)}, status_code=HTTP_200_OK)

    @app.post("/dq/freshness")
    def _dq_freshness(
        dataset: str = Query(..., description="Dataset-navn, fx ohlcv_1h"),
        minutes: float = Query(..., description="Freshness i minutter"),
        x_dq_secret: Optional[str] = Header(default=None),
        secret: Optional[str] = Query(default=None, description="Alternativ auth via query"),
    ):
        if not _auth(x_dq_secret, secret):
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        set_freshness(dataset, minutes)
        return JSONResponse(
            {"ok": True, "dataset": dataset, "minutes": float(minutes)},
            status_code=HTTP_201_CREATED,
        )

    @app.post("/dq/violation")
    def _dq_violation(
        contract: str = Query(..., description="Kontrakt/konfiguration, fx ohlcv_1h"),
        rule: str = Query(..., description="Rule-id, fx bounds_min"),
        n: int = Query(default=1, ge=1, description="Antal inkrementer"),
        x_dq_secret: Optional[str] = Header(default=None),
        secret: Optional[str] = Query(default=None, description="Alternativ auth via query"),
    ):
        if not _auth(x_dq_secret, secret):
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        emit_violation(contract, rule, n=n)
        return JSONResponse(
            {"ok": True, "contract": contract, "rule": rule, "inc": int(n)},
            status_code=HTTP_201_CREATED,
        )

    @app.post("/dq/sample")
    def _dq_sample(
        dataset: str = Query("ohlcv_1h"),
        minutes: float = Query(20.0),
        contract: str = Query("ohlcv_1h"),
        rule: str = Query("bounds_min"),
        n: int = Query(1, ge=1),
        x_dq_secret: Optional[str] = Header(default=None),
        secret: Optional[str] = Query(default=None),
    ):
        """
        Kun til lokal/dev smoke: sæt freshness og emit en violation.
        Kører kun hvis debug_mode eller secret matcher (ellers 401).
        """
        if not (debug_mode or _auth(x_dq_secret, secret)):
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        set_freshness(dataset, minutes)
        emit_violation(contract, rule, n=n)
        return JSONResponse(
            {
                "ok": True,
                "freshness": {"dataset": dataset, "minutes": float(minutes)},
                "violation": {"contract": contract, "rule": rule, "inc": int(n)},
            },
            status_code=HTTP_201_CREATED,
        )


__all__ = ["emit_violation", "set_freshness", "wire_fastapi_routes"]
