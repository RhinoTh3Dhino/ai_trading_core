# bot/__init__.py
"""
Package init for 'bot'.

Formål
------
- Undgå dobbelt-registrering af Prometheus-metrikker ved processtart (CI/uvicorn).
- Registrér IKKE live-connector metrikker ved import.
- Registrér IKKE core-metrikker automatisk som default (kan tændes via env).

Miljøvariable
-------------
- METRICS_EAGER=0/1       (default: 0)  → hvis 1, registreres core-metrikker ved import.
- METRICS_EAGER_LIVE=0/1  (default: 0)  → hvis 1, registreres live-metrikker ved import
                                          (brug KUN i processer der ellers ikke gør det).
- METRICS_BOOTSTRAP=0/1   (default: 0)  → hvis METRICS_EAGER_LIVE=1, kaldes bootstrap_core_metrics().
"""

from __future__ import annotations
import os


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


# --- Core-metrikker (valgfrit, default OFF) -----------------------------------
# Bemærk: Disse kan overlappe live-connectorens navne i nogle setups.
# Vi lader dem derfor være slukket som default, så runner ikke crasher i CI.
if _env_on("METRICS_EAGER", "0"):
    try:
        # Skal være idempotent inde i metrics_core
        from .metrics_core import init_core_metrics  # type: ignore
        init_core_metrics()
    except Exception:
        # Må aldrig vælte import af pakken
        pass


# --- Live-connector metrikker (FRIVILLIGT, default OFF) -----------------------
# Kald IKKE dette ved import som standard. Runner/startup håndterer det selv.
if _env_on("METRICS_EAGER_LIVE", "0"):
    try:
        from .live_connector import metrics as _m  # type: ignore
        try:
            _m.ensure_registered()  # idempotent i modulet, men vi holder det slukket her
        except Exception:
            pass

        if _env_on("METRICS_BOOTSTRAP", "0"):
            try:
                if hasattr(_m, "bootstrap_core_metrics"):
                    _m.bootstrap_core_metrics()
            except Exception:
                pass
    except Exception:
        # Må ikke blokere import af bot-pakken
        pass
