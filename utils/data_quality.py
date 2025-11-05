# utils/data_quality.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if spec.dtype == "float":
        return pd.to_numeric(series, errors="coerce")
    if spec.dtype == "bool":
        return series.astype("boolean")
    return series.astype("string")


def validate(df: pd.DataFrame, contract: DataContract) -> DQReport:
    issues: Dict[str, Any] = {}

    # 1) Schema/kolonner
    missing = [c for c in contract.required_cols if c not in df.columns]
    if missing:
        issues["missing_columns"] = missing

    # 2) Typer + null-rate + bounds
    typed: Dict[str, pd.Series] = {}
    for col, spec in contract.required_cols.items():
        if col not in df.columns:
            continue
        s = _coerce_dtype(df[col], spec)
        typed[col] = s
        null_rate = _rate(int(s.isna().sum()), len(s))
        if not spec.allow_null and null_rate > 0:
            issues.setdefault("null_rate", {})[col] = round(null_rate, 6)

        if spec.dtype in {"float", "int"} and s.notna().any():
            # Bemærk: Int64 (pandas NA) → cast til float for min/max
            vmin, vmax = float(s.min()), float(s.max())
            if spec.min_val is not None and vmin < spec.min_val:
                issues.setdefault("bounds_min", {})[col] = vmin
            if spec.max_val is not None and vmax > spec.max_val:
                issues.setdefault("bounds_max", {})[col] = vmax

    # 3) Rækker/duplikater
    if len(df) < contract.min_rows:
        issues["min_rows"] = len(df)

    if contract.key_cols:
        dup_rate = _rate(int(df.duplicated(contract.key_cols).sum()), len(df))
        if dup_rate > contract.max_dup_rate:
            issues["dup_rate"] = round(dup_rate, 6)

    # 4) Global null-rate over required cols (valgfri ekstra regel)
    subset = [c for c in contract.required_cols.keys() if c in df.columns]
    if subset:
        total_nulls = int(df[subset].isna().sum().sum())
        total_cells = int(len(df) * len(subset))
        global_null_rate = _rate(total_nulls, total_cells)
        if global_null_rate > contract.max_null_rate:
            issues["global_null_rate"] = round(global_null_rate, 6)

    return DQReport(ok=(len(issues) == 0), issues=issues)
