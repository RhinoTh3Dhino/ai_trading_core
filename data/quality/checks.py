# data/quality/checks.py
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None  # type: ignore

# --------------------------------------------------------------------------
# Konfiguration (kan sættes via .env / compose environment)
# --------------------------------------------------------------------------
DQ_SECRET: str = os.getenv("DQ_SHARED_SECRET", "change-me-long-random")
DQ_URL: str = os.getenv("DQ_URL", "http://localhost:8000").rstrip("/")

_DQ_TIMEOUT_SEC: float = float(os.getenv("DQ_TIMEOUT", "5"))
_DQ_RETRIES: int = int(os.getenv("DQ_RETRIES", "3"))
_DQ_BACKOFF: float = float(os.getenv("DQ_RETRY_BACKOFF", "0.5"))

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# --------------------------------------------------------------------------
# HTTP session m. retries/backoff
# --------------------------------------------------------------------------
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


def _post_violation(
    contract: str,
    rule: str,
    n: int = 1,
    *,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    sess = session or _session
    headers = {"X-Dq-Secret": DQ_SECRET}
    params = {"contract": contract, "rule": rule, "n": int(n)}
    url = f"{DQ_URL}/dq/violation"
    return sess.post(url, params=params, headers=headers, timeout=_DQ_TIMEOUT_SEC)


# --------------------------------------------------------------------------
# Offentligt API (bagudkompatibelt)
# --------------------------------------------------------------------------
def inc_violation(
    contract: str,
    rule: str,
    n: int = 1,
    *,
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> bool:
    """
    Inkrementér en DQ-violation-tæller i live_connector.

    Returns:
        True ved HTTP 2xx, ellers False (eller exception hvis raise_on_error=True).
    """
    try:
        contract = (contract or "").strip()
        rule = (rule or "").strip()
        if not contract:
            raise ValueError("contract må ikke være tom")
        if not rule:
            raise ValueError("rule må ikke være tom")
        if n <= 0:
            return True  # intet at gøre

        resp = _post_violation(contract, rule, n=n, session=session)
        resp.raise_for_status()
        return True
    except Exception as e:
        if raise_on_error:
            raise
        log.warning(
            "[DQ] violation post failed (contract=%s, rule=%s, n=%s): %s", contract, rule, n, e
        )
        return False


def check_global_null_rate(
    df: pd.DataFrame,
    contract: str,
    threshold: float = 0.05,
    *,
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> float:
    """
    Mål null-rate på tværs af hele DataFrame. Inkrementér 'global_null_rate' hvis rate > threshold.

    Returns:
        Den beregnede null-rate (0.0 hvis df er tom).
    """
    try:
        total = int(df.size)
        if total == 0:
            return 0.0
        nulls = int(df.isna().sum().sum())
        rate = nulls / float(total)
        if rate > threshold:
            inc_violation(
                contract, "global_null_rate", 1, raise_on_error=raise_on_error, session=session
            )
        return rate
    except Exception as e:
        if raise_on_error:
            raise
        log.warning("[DQ] check_global_null_rate failed: %s", e)
        return 0.0


# --------------------------------------------------------------------------
# Udvidede hjælpecases (valgfri, men nyttige i praksis)
# Navngivning af rules er konsekvent og lav-kardinalitet.
# --------------------------------------------------------------------------
def check_column_null_rate(
    df: pd.DataFrame,
    column: str,
    contract: str,
    threshold: float = 0.05,
    *,
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> float:
    """
    Null-rate for én kolonne. Rule-navn: 'null_rate:<col>'.
    """
    try:
        if column not in df.columns:
            # Manglende kolonne er i sig selv et brud på schema — rapportér særskilt
            inc_violation(
                contract,
                f"missing_column:{column}",
                1,
                raise_on_error=raise_on_error,
                session=session,
            )
            return 1.0
        s = df[column]
        total = int(s.shape[0])
        if total == 0:
            return 0.0
        rate = float(s.isna().sum()) / float(total)
        if rate > threshold:
            inc_violation(
                contract, f"null_rate:{column}", 1, raise_on_error=raise_on_error, session=session
            )
        return rate
    except Exception as e:
        if raise_on_error:
            raise
        log.warning("[DQ] check_column_null_rate failed (col=%s): %s", column, e)
        return 0.0


def check_value_range(
    df: pd.DataFrame,
    column: str,
    contract: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    frac_threshold: float = 0.0,
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> float:
    """
    Tjek at værdier i kolonne ligger inden for [min_value, max_value].
    Hvis andelen udenfor intervallet > frac_threshold, inkrementeres 'range_violation:<col>'.

    Returns:
        Fraktionen af rækker udenfor intervallet (0..1). Returnerer 0.0 hvis kolonnen mangler eller er tom.
    """
    try:
        if column not in df.columns:
            inc_violation(
                contract,
                f"missing_column:{column}",
                1,
                raise_on_error=raise_on_error,
                session=session,
            )
            return 1.0 if frac_threshold == 0 else 0.0

        s = df[column].dropna()
        n = int(s.shape[0])
        if n == 0:
            return 0.0

        mask = pd.Series(False, index=s.index)
        if min_value is not None:
            mask = mask | (s < min_value)
        if max_value is not None:
            mask = mask | (s > max_value)

        out_frac = float(mask.sum()) / float(n)
        if out_frac > frac_threshold:
            inc_violation(
                contract,
                f"range_violation:{column}",
                1,
                raise_on_error=raise_on_error,
                session=session,
            )
        return out_frac
    except Exception as e:
        if raise_on_error:
            raise
        log.warning("[DQ] check_value_range failed (col=%s): %s", column, e)
        return 0.0


def check_duplicates(
    df: pd.DataFrame,
    contract: str,
    *,
    subset: Optional[Iterable[str]] = None,
    allow_n: int = 0,
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> int:
    """
    Tæl duplikatrækker (globalt eller pr. subset). Hvis antal > allow_n, inkrementér 'duplicates'.

    Returns:
        Antal duplikater (int).
    """
    try:
        dup_mask = df.duplicated(subset=list(subset) if subset else None, keep=False)
        count = int(dup_mask.sum())
        if count > allow_n:
            inc_violation(contract, "duplicates", 1, raise_on_error=raise_on_error, session=session)
        return count
    except Exception as e:
        if raise_on_error:
            raise
        log.warning("[DQ] check_duplicates failed: %s", e)
        return 0


def check_datetime_monotonic(
    s: pd.Series,
    contract: str,
    *,
    allow_equal: bool = False,
    rule_name: str = "ts_monotonic",
    raise_on_error: bool = False,
    session: Optional[requests.Session] = None,
) -> bool:
    """
    Tjek at en tidsserie er strengt (eller svagt) stigende. Inkrementér 'ts_monotonic' ved brud.

    Args:
        s: Pandas Series med datetime64 (eller noget der kan konverteres).
        allow_equal: Tillad lighed (<=) hvis True.
        rule_name: Tilpas rule-navn hvis ønsket.

    Returns:
        True hvis monotoni-består, ellers False.
    """
    try:
        if not pd.api.types.is_datetime64_any_dtype(s):
            # prøv at konvertere
            s = pd.to_datetime(s, errors="coerce")

        # fjern NaT først (de kan kontrolleres med null-rate separat)
        s = s.dropna()
        if s.shape[0] <= 1:
            return True

        diffs = s.values[1:] - s.values[:-1]
        # numpy timedelta64, tjek >0 eller >=0
        if allow_equal:
            ok = (diffs >= pd.Timedelta(0)).all()
        else:
            ok = (diffs > pd.Timedelta(0)).all()

        if not ok:
            inc_violation(contract, rule_name, 1, raise_on_error=raise_on_error, session=session)
        return bool(ok)

    except Exception as e:
        if raise_on_error:
            raise
        log.warning("[DQ] check_datetime_monotonic failed: %s", e)
        return False
