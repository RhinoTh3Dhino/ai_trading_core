#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class DQIssue:
    contract: str
    rule: str
    details: str
    count: int = 1


OK = 0
WARN = 10
FAIL = 20


def _read_table(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "parquet":
        return pd.read_parquet(path, columns=columns)
    if ext in {"csv", "txt"}:
        return pd.read_csv(path, usecols=columns)
    raise SystemExit(f"Unsupported file type: {ext}")


def _pct(x: int, denom: int) -> float:
    return 0.0 if denom <= 0 else (100.0 * x / denom)


def _post_violation(endpoint: str, secret: str, issue: DQIssue, n: int = 1) -> None:
    """Best-effort POST til live-connectorens /dq/violation. Importér requests lazy for
    at undgå dependency i helt minimale miljøer."""
    try:
        import requests  # type: ignore
    except Exception:
        return
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/dq/violation",
            headers={"X-Dq-Secret": secret},
            params={"contract": issue.contract, "rule": issue.rule, "n": max(1, n)},
            timeout=5,
        )
        r.raise_for_status()
    except Exception:
        pass


def check_basic_contract(
    df: pd.DataFrame,
    contract: str = "ohlcv_1h",
    ts_col: str = "ts",
    price_col: str = "close",
    volume_col: str = "volume",
    max_nan_pct: float = 1.0,
    require_monotonic_ts: bool = True,
    min_price: float = 1e-9,
    min_volume: float = 0.0,
) -> List[DQIssue]:
    issues: List[DQIssue] = []
    # 1) Basisskema
    for col in (ts_col, price_col, volume_col):
        if col not in df.columns:
            issues.append(DQIssue(contract, "schema_missing", f"missing={col}"))
    if issues:
        return issues

    n = len(df)
    if n == 0:
        issues.append(DQIssue(contract, "empty_dataset", "no rows"))
        return issues

    # 2) NaN-rate
    nan_cols: List[Tuple[str, float]] = []
    for col in (price_col, volume_col):
        c = df[col].isna().sum()
        p = _pct(c, n)
        if p > max_nan_pct:
            issues.append(DQIssue(contract, "nan_rate_excess", f"col={col}, pct={p:.2f}%"))
        if p > 0:
            nan_cols.append((col, p))

    # 3) Monotone timestamps
    if require_monotonic_ts:
        ts = pd.Index(df[ts_col].astype("int64"))
        if not ts.is_monotonic_increasing:
            issues.append(DQIssue(contract, "ts_not_monotonic", "timestamps not sorted asc"))

        # Duplikate ts?
        dup = df.duplicated(subset=[ts_col]).sum()
        if dup > 0:
            issues.append(DQIssue(contract, "ts_duplicate", f"duplicates={dup}"))

    # 4) Bounds
    if (df[price_col] < min_price).any():
        cnt = int((df[price_col] < min_price).sum())
        issues.append(DQIssue(contract, "bounds_min_price", f"<{min_price} count={cnt}", cnt))
    if (df[volume_col] < min_volume).any():
        cnt = int((df[volume_col] < min_volume).sum())
        issues.append(DQIssue(contract, "bounds_min_volume", f"<{min_volume} count={cnt}", cnt))

    # 5) Ikke-numeriske outliers (enkelt, konservativ Z-score på log-returns)
    #    Kun hvis nok rækker.
    if n >= 50:
        try:
            close = pd.to_numeric(df[price_col], errors="coerce")
            lr = (close / close.shift(1)).apply(
                lambda v: math.log(v) if v and v > 0 else float("nan")
            )
            s = lr.dropna()
            if len(s) >= 30:
                z = (s - s.mean()) / (s.std(ddof=1) + 1e-12)
                extreme = (z.abs() > 10).sum()  # meget høj tærskel (kun åbenlyse spikes)
                if extreme > 0:
                    issues.append(
                        DQIssue(
                            contract,
                            "returns_extreme_spikes",
                            f"count={int(extreme)}",
                            int(extreme),
                        )
                    )
        except Exception:
            pass

    return issues


def _print_report(issues: Iterable[DQIssue], print_report: bool) -> None:
    if not print_report:
        return
    print("\n# Data Quality Report\n")
    rows = list(issues)
    if not rows:
        print("✔ Ingen problemer fundet.")
        return
    for it in rows:
        print(f"- [{it.contract}] {it.rule} :: {it.details} (n={it.count})")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ad-hoc Data Quality checker")
    p.add_argument("--dataset", required=True, help="Path til .parquet eller .csv")
    p.add_argument("--contract", default="ohlcv_1h")
    p.add_argument("--ts-col", default="ts")
    p.add_argument("--price-col", default="close")
    p.add_argument("--volume-col", default="volume")
    p.add_argument("--max-nan-pct", type=float, default=1.0)
    p.add_argument("--no-monotonic", action="store_true", help="Deaktiver ts-monotonic check")
    p.add_argument("--min-price", type=float, default=1e-9)
    p.add_argument("--min-volume", type=float, default=0.0)
    p.add_argument("--print-report", action="store_true")
    p.add_argument("--post-endpoint", default="", help="fx http://localhost:8000")
    p.add_argument("--post-secret", default="", help="værdi til X-Dq-Secret")
    p.add_argument(
        "--fail-on-issues", action="store_true", help="Exit != 0 hvis der findes problemer"
    )
    args = p.parse_args(argv)

    df = _read_table(args.dataset, columns=[args.ts_col, args.price_col, args.volume_col])
    issues = check_basic_contract(
        df=df,
        contract=args.contract,
        ts_col=args.ts_col,
        price_col=args.price_col,
        volume_col=args.volume_col,
        max_nan_pct=args.max_nan_pct,
        require_monotonic_ts=not args.no_monotonic,
        min_price=args.min_price,
        min_volume=args.min_volume,
    )

    # Best-effort POST til live-connector (så alerts kan fyre i Prometheus)
    if args.post_endpoint and args.post_secret:
        for it in issues:
            _post_violation(args.post_endpoint, args.post_secret, it, n=max(1, it.count))

    _print_report(issues, args.print_report)

    if args.fail_on_issues and issues:
        return FAIL
    return OK


if __name__ == "__main__":
    sys.exit(main())
