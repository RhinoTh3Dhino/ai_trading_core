# bot/__init__.py
"""
Package init, der sikrer at Prometheus-metrikker for feed & features
er registreret idempotent i den proces, der kører appen.

Design:
- Trin 1: Registrér "core" metrikker via bot.metrics_core (idempotent).
- Trin 2: Hvis live_connector.metrics findes, kør ensure_registered()
          og (valgfrit) bootstrap_core_metrics().

Begge trin er failsafe: Exceptions sluges, så importen af 'bot' aldrig
vælter app/tests.

Env-styring:
- METRICS_EAGER=0/1     (default: 1)  → slå eager init helt til/fra
- METRICS_BOOTSTRAP=0/1 (default: 0)  → om bootstrap skal køres (observe(0) m.m.)
"""

from __future__ import annotations
import os

def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}

# Idempotens-flag pr. proces
_CORE_INIT_DONE = False
_METRICS_ENSURED = False
_METRICS_BOOTSTRAPPED = False

if _env_on("METRICS_EAGER", "1"):
    # Trin 1: Registrér core-metrics fra bot.metrics_core (idempotent)
    if not _CORE_INIT_DONE:
        try:
            from .metrics_core import init_core_metrics  # type: ignore
            init_core_metrics()   # bør selv være idempotent
            _CORE_INIT_DONE = True
        except Exception:
            # Må ikke blokere import af bot-pakken
            pass

    # Trin 2: Brug live_connector.metrics hvis det findes
    try:
        from .live_connector import metrics as _m  # type: ignore

        if not _METRICS_ENSURED:
            try:
                _m.ensure_registered()
                _METRICS_ENSURED = True
            except Exception:
                pass

        # Bootstrap kun hvis ønsket via env og ikke allerede gjort
        if _env_on("METRICS_BOOTSTRAP", "0") and not _METRICS_BOOTSTRAPPED:
            try:
                # Skal gerne være idempotent; vi beskytter alligevel med flag
                if hasattr(_m, "bootstrap_core_metrics"):
                    _m.bootstrap_core_metrics()
                _METRICS_BOOTSTRAPPED = True
            except Exception:
                pass

    except Exception:
        # Må ikke blokere import af bot-pakken
        pass
