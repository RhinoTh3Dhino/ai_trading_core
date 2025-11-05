# data/ingest/freshness_reporter.py
from __future__ import annotations

import logging
import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

try:
    # Retry er tilgængelig via urllib3 i requests
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None  # type: ignore

# --- Konfiguration fra env ----------------------------------------------------
DQ_SECRET: str = os.getenv("DQ_SHARED_SECRET", "change-me-long-random")
DQ_URL: str = os.getenv("DQ_URL", "http://localhost:8000").rstrip("/")

# Tuning (kan overskrives i CI/drift)
_DQ_TIMEOUT_SEC: float = float(os.getenv("DQ_TIMEOUT", "5"))
_DQ_RETRIES: int = int(os.getenv("DQ_RETRIES", "3"))
_DQ_BACKOFF: float = float(os.getenv("DQ_RETRY_BACKOFF", "0.5"))

# Logger
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# --- Session med (valgfri) retries -------------------------------------------
def _build_session() -> requests.Session:
    s = requests.Session()
    if Retry is None or _DQ_RETRIES <= 0:
        return s

    retry = Retry(
        total=_DQ_RETRIES,
        connect=_DQ_RETRIES,
        read=_DQ_RETRIES,
        backoff_factor=_DQ_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST", "GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


_session = _build_session()


# --- Public API ---------------------------------------------------------------
def report_freshness(
    dataset: str,
    minutes_since_update: float,
    *,
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> bool:
    """
    Rapporter dataset-freshness til live_connector DQ-endpoint.

    Args:
        dataset: Navn/label for dataset (fx 'ohlcv_1h').
        minutes_since_update: Antal minutter siden seneste succesfulde write/rotation.
        raise_on_error: Hvis True kastes exception på fejl, ellers logges og returneres False.
        session: Valgfri requests.Session (ellers bruges modul-deriveret session med retries).

    Returns:
        True hvis posten lykkes (HTTP 2xx), ellers False (eller exception hvis raise_on_error=True).
    """
    try:
        ds = (dataset or "").strip()
        if not ds:
            raise ValueError("dataset må ikke være tom")

        minutes = float(minutes_since_update)
        if minutes < 0:
            raise ValueError("minutes_since_update skal være >= 0")

        sess = session or _session
        headers = {"X-Dq-Secret": DQ_SECRET}
        params = {"dataset": ds, "minutes": minutes}

        url = f"{DQ_URL}/dq/freshness"
        resp = sess.post(url, params=params, headers=headers, timeout=_DQ_TIMEOUT_SEC)
        resp.raise_for_status()
        return True

    except Exception as e:
        if raise_on_error:
            raise
        log.warning(
            "[DQ] freshness report failed (dataset=%s, minutes=%s): %s",
            dataset,
            minutes_since_update,
            e,
        )
        return False


# --- CLI (valgfri) -----------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Report DQ freshness to live_connector.")
    p.add_argument("--dataset", required=True, help="Dataset navn (fx ohlcv_1h)")
    p.add_argument(
        "--minutes", required=True, type=float, help="Minutter siden seneste opdatering (>=0)"
    )
    p.add_argument(
        "--raise", dest="raise_on_error", action="store_true", help="Kast exception ved fejl"
    )
    args = p.parse_args()

    ok = report_freshness(args.dataset, args.minutes, raise_on_error=args.raise_on_error)
    print({"ok": ok, "dataset": args.dataset, "minutes": float(args.minutes)})
