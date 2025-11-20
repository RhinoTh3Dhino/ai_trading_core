# bot/persist/ohlcv_writer.py
from __future__ import annotations

"""
Persist OHLCV (1h) med DQ-validering og Prometheus-emittering.

Brug:
  python -m bot.persist.ohlcv_writer --src data/ohlcv_1h.csv --out outputs/ohlcv_1h.parquet --print-report
  # eller
  python bot/persist/ohlcv_writer.py --src tests/_data/ohlcv_good.parquet --out outputs/ohlcv_1h.parquet

Exit codes:
  0 = OK
  1 = DQ-fejl (hvis --fail-on-dq) eller uventet exception
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from utils.data_contracts import ColumnSpec, DataContract
from utils.dq_wiring import dq_check_and_emit

# --------------------------------------------------------------------------------------
# Konfiguration / kontrakt
# --------------------------------------------------------------------------------------

LOG = logging.getLogger("persist.ohlcv_writer")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DATASET_NAME = "ohlcv_1h"

CONTRACT_OHLCV_1H = DataContract(
    name=DATASET_NAME,
    required_cols={
        "timestamp": ColumnSpec("datetime", allow_null=False),
        "open": ColumnSpec("float", min_val=0),
        "high": ColumnSpec("float", min_val=0),
        "low": ColumnSpec("float", min_val=0),
        "close": ColumnSpec("float", min_val=0),
        "volume": ColumnSpec("float", min_val=0),
    },
    key_cols=("timestamp",),
    min_rows=100,
    max_dup_rate=0.02,
    max_null_rate=0.01,
)


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------


def _load_df(src: Path) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"Input findes ikke: {src}")
    suf = src.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(src)
    if suf == ".csv":
        return pd.read_csv(src)
    raise ValueError(f"Ukendt inputformat: {suf} (understøtter .csv, .parquet)")


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Gør 'timestamp' tz-aware (UTC). Tåler allerede-UTC.
    """
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s


def _freshness_minutes_from_df(df: pd.DataFrame) -> Optional[float]:
    """
    Freshness = now_utc - max(timestamp) i minutter. Returnerer None, hvis mangler.
    """
    if "timestamp" not in df.columns or df.empty:
        return None
    ts = _to_utc_datetime(df["timestamp"])
    if ts.isna().all():
        return None
    ts_max = ts.max()
    now = pd.Timestamp.utcnow()
    delta = now - ts_max
    minutes = max(delta.total_seconds() / 60.0, 0.0)
    return float(minutes)


def _freshness_minutes_from_file(src: Path) -> Optional[float]:
    try:
        mtime = src.stat().st_mtime  # sekunder (lokal systemtid)
        ts = pd.Timestamp.utcfromtimestamp(mtime)
        now = pd.Timestamp.utcnow()
        return float(max((now - ts).total_seconds() / 60.0, 0.0))
    except Exception:
        return None


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Core persist
# --------------------------------------------------------------------------------------


@dataclass
class PersistResult:
    ok: bool
    issues: dict
    freshness_minutes: Optional[float]
    out_path: Path
    rows: int


def persist_ohlcv_1h(
    src: Path, out_path: Path, fail_on_dq: bool = True, print_report: bool = False
) -> PersistResult:
    """
    Loader src, beregner freshness, kører DQ, emitter metrics, skriver Parquet.
    """
    df = _load_df(src)

    # Normaliser timestamp til UTC (kun hvis kolonnen findes)
    if "timestamp" in df.columns:
        df["timestamp"] = _to_utc_datetime(df["timestamp"])

    # Freshness prioritet: seneste timestamp i data → fallback: filens mtime
    freshness = _freshness_minutes_from_df(df)
    if freshness is None:
        freshness = _freshness_minutes_from_file(src)

    ok, issues = dq_check_and_emit(
        df=df,
        contract=CONTRACT_OHLCV_1H,
        dataset_name=DATASET_NAME,
        minutes_since_last_update=freshness if freshness is not None else None,
    )

    if print_report:
        import json

        LOG.info("DQ-rapport: %s", json.dumps({"ok": ok, "issues": issues}, ensure_ascii=False))

    if not ok and fail_on_dq:
        # Emit er allerede sket i dq_check_and_emit – vi stopper batchen her
        raise ValueError(f"DQ failed for {DATASET_NAME}: {issues}")

    # Skriv altid output (selv ved DQ-fejl) hvis du kører i “log-only”-mode
    _ensure_parent_dir(out_path)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    return PersistResult(
        ok=ok, issues=issues, freshness_minutes=freshness, out_path=out_path, rows=len(df)
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("persist_ohlcv_1h")
    p.add_argument("--src", required=True, help="Input CSV/Parquet med OHLCV (1h)")
    p.add_argument("--out", required=True, help="Output Parquet-fil")
    p.add_argument(
        "--no-fail-on-dq",
        action="store_true",
        help="Fejl ikke pipeline ved DQ-brud (logger kun og fortsætter)",
    )
    p.add_argument("--print-report", action="store_true", help="Log JSON-rapport for DQ")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    src = Path(args.src)
    out = Path(args.out)
    fail_on_dq = not args.no_fail_on_dq

    try:
        res = persist_ohlcv_1h(src, out, fail_on_dq=fail_on_dq, print_report=args.print_report)
        LOG.info(
            "Persist OK=%s rows=%s freshness_min=%s out=%s issues=%s",
            res.ok,
            res.rows,
            res.freshness_minutes,
            res.out_path,
            res.issues or {},
        )
        # Exitkode: 0 hvis OK eller no-fail-mode; 1 hvis DQ-fejl og fail-mode
        return 0 if (res.ok or not fail_on_dq) else 1
    except Exception as e:
        LOG.exception("Persist fejlede: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
