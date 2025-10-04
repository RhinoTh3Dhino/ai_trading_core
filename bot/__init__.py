# bot/__init__.py
"""
Package init for 'bot'.

Formål
------
- Sørg for, at generiske/core Prometheus-metrikker er tilgængelige tidligt.
- Undgå dobbelt-registrering af live-connector metrikker, som ellers kan
  crashe processer i CI/uvicorn med "Duplicated timeseries in CollectorRegistry".

Design
------
- Vi registrerer KUN 'core' metrikker her via bot.metrics_core (idempotent).
- Live-connector-metrikker (bot.live_connector.metrics.ensure_registered)
  kaldes IKKE her. De kaldes i de komponenter, der faktisk starter live-
  connectoren (runner/engine), så de ikke bliver dobbelt-registreret.

Env-styring
-----------
- METRICS_EAGER=0/1     (default: 1)  → om core-metrikker registreres ved import.
- METRICS_EAGER_LIVE=0/1 (default: 0) → (valgfrit) tving eager registrering af
                                        live-connector-metrikker her (brug kun
                                        hvis du VED, at processen ikke også
                                        gør det senere).
- METRICS_BOOTSTRAP=0/1 (default: 0)  → hvis METRICS_EAGER_LIVE=1, kan vi
                                        (valgfrit) kalde bootstrap_core_metrics().
"""

from __future__ import annotations
import os


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


# --- Trin 1: Core-metrikker (idempotent, ufarligt) ----------------------------
if _env_on("METRICS_EAGER", "1"):
    try:
        # Denne import udfører registreringen idempotent
        from .metrics_core import init_core_metrics  # type: ignore
        init_core_metrics()
    except Exception:
        # MÅ IKKE vælte import af 'bot' – tests/app skal altid kunne starte
        pass


# --- Trin 2: (FRIVILLIGT) Live-connector metrikker ----------------------------
# Som udgangspunkt gør vi IKKE dette her, for at undgå dobbelt-registrering.
# Brug KUN denne blok, hvis du specifikt sætter METRICS_EAGER_LIVE=1 i en
# proces, som ellers ikke selv kalder ensure_registered() senere.
if _env_on("METRICS_EAGER_LIVE", "0"):
    try:
        from .live_connector import metrics as _m  # type: ignore

        # Registrér live-metrikker idempotent
        try:
            _m.ensure_registered()
        except Exception:
            pass

        # (Valgfrit) Bootstrap – fx så histogram-buckets er synlige straks
        if _env_on("METRICS_BOOTSTRAP", "0"):
            try:
                if hasattr(_m, "bootstrap_core_metrics"):
                    _m.bootstrap_core_metrics()
            except Exception:
                pass

    except Exception:
        # Må ikke blokere import af bot-pakken
        pass
