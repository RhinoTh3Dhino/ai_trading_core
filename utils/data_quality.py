# utils/data_quality.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from .data_contracts import ColumnSpec, DataContract


@dataclass
class DQReport:
    ok: bool
    issues: Dict[str, Any]  # {rule: details}


def _rate(n: int, d: int) -> float:
    return 0.0 if d == 0 else float(n) / float(d)


def _coerce_dtype(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    if spec.dtype == "datetime":
        return pd.to_datetime(series, errors="coerce", utc=True)
    if spec.dtype == "int":
        # Bevar NA vha. pandas' nullable Int64
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if spec.dtype == "float":
        return pd.to_numeric(series, errors="coerce")
    if spec.dtype == "bool":
        return series.astype("boolean")
    # default → string (nullable)
    return series.astype("string")


def validate(df: pd.DataFrame, contract: DataContract) -> DQReport:
    issues: Dict[str, Any] = {}

    have_cols = set(df.columns)
    req_cols = list(contract.required_cols.keys())

    # 1) Schema/kolonner
    missing: List[str] = [c for c in req_cols if c not in have_cols]
    if missing:
        issues["missing_columns"] = missing

    # 2) Typer + null-rate + bounds (kun for kolonner der faktisk findes)
    typed: Dict[str, pd.Series] = {}
    for col, spec in contract.required_cols.items():
        if col not in have_cols:
            continue
        s = _coerce_dtype(df[col], spec)
        typed[col] = s

        # Null-rate for kolonnen
        null_rate = _rate(int(s.isna().sum()), len(s))
        if getattr(spec, "allow_null", False) is False and null_rate > 0:
            issues.setdefault("null_rate", {})[col] = round(null_rate, 6)

        # Bounds for taltyper
        if spec.dtype in {"float", "int"} and s.notna().any():
            vmin, vmax = float(s.min()), float(s.max())
            if spec.min_val is not None and vmin < spec.min_val:
                issues.setdefault("bounds_min", {})[col] = vmin
            if spec.max_val is not None and vmax > spec.max_val:
                issues.setdefault("bounds_max", {})[col] = vmax

    # 3) Rækker/duplikater
    if len(df) < getattr(contract, "min_rows", 0):
        issues["min_rows"] = len(df)

    if getattr(contract, "key_cols", ()):
        # Beskyt mod KeyError: beregn kun dup-rate hvis alle key-kolonner findes
        key_missing = [k for k in contract.key_cols if k not in have_cols]
        if key_missing:
            # Rapportér pænt at key-kolonner mangler, men kast ingen fejl
            issues.setdefault("missing_key_cols", key_missing)
        else:
            dup_rate = _rate(int(df.duplicated(contract.key_cols).sum()), len(df))
            if dup_rate > getattr(contract, "max_dup_rate", 0.0):
                issues["dup_rate"] = round(dup_rate, 6)

    # 4) Global null-rate over required cols (kun for eksisterende kolonner)
    subset = [c for c in req_cols if c in have_cols]
    if subset:
        total_nulls = int(df[subset].isna().sum().sum())
        total_cells = int(len(df) * len(subset))
        global_null_rate = _rate(total_nulls, total_cells)
        if global_null_rate > getattr(contract, "max_null_rate", 1.0):
            issues["global_null_rate"] = round(global_null_rate, 6)

    return DQReport(ok=(len(issues) == 0), issues=issues)
