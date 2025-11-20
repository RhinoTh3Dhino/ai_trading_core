# tools/rotate_outputs.py
"""
Fase 4 — Persistens & Filhygiejne (rotation + kompaktering)

Eksempler:
  python -m tools.rotate_outputs
  python -m tools.rotate_outputs --compact --symbol BTCUSDT --interval 1m
  python -m tools.rotate_outputs --compact --symbol BTCUSDT --interval 1m --date 2025-10-11
  python -m tools.rotate_outputs --compact --symbol BTCUSDT --interval 1m --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from config.config import PERSIST
from utils.artifacts import compact_day, partition_path, rotate_dir

# ------------------------------
# Legacy mappe-rotation (beholdt)
# ------------------------------
TARGETS: Dict[str, Tuple[int, str]] = {
    "outputs/feature_data": (3, r".*\.(csv|parquet)$"),  # små tal for nem test
    "outputs/labels": (3, r".*\.npy$"),
    "outputs/models": (2, r".*\.(keras|h5)$"),
    "outputs/backtests": (3, r".*\.(csv|png)$"),
    "outputs/metrics": (5, r".*\.json$"),
    "outputs/charts": (5, r".*\.png$"),
    "outputs/logs": (5, r".*\.log$"),
    "archives": (5, r".*\.zip$"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotation og kompaktering af outputs.")
    p.add_argument("--dry-run", action="store_true", help="Vis hvad der sker, men skriv ikke.")
    p.add_argument("--compact", action="store_true", help="Kompaktér dags-partitioner.")
    p.add_argument("--symbol", type=str, help="F.eks. BTCUSDT")
    p.add_argument("--interval", type=str, help="F.eks. 1m, 5m, 1h")
    p.add_argument("--date", type=str, help="YYYY-MM-DD (udeladt = i dag, UTC).")
    p.add_argument("--skip-legacy", action="store_true", help="Spring legacy mappe-rotation over.")
    return p.parse_args()


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _print(msg: str) -> None:
    print(msg, flush=True)


def _run_legacy_rotation() -> None:
    for d, (keep, pat) in TARGETS.items():
        rotate_dir(d, keep=keep, pattern=pat)
    _print("✅ Legacy rotation gennemført")


def _run_compact(symbol: str, interval: str, date_str: str, dry_run: bool) -> int:
    day_dir = partition_path(symbol, interval, date_str)
    if dry_run:
        _print(
            f"[DRY-RUN] Ville kompaktere: {day_dir} "
            f"(features_version={PERSIST['FEATURES_VERSION']})"
        )
        return 0

    stats = compact_day(day_dir, expected_features_version=PERSIST["FEATURES_VERSION"])
    _print(
        "✅ Kompaktering fuldført | "
        f"in_files={stats['in_files']} -> out_files={stats['out_files']}, "
        f"in_rows={stats['in_rows']} -> out_rows={stats['out_rows']}, "
        f"dropped_dups={stats['dropped_dups']}"
    )
    return 0


def main() -> int:
    ns = _parse_args()

    if not ns.skip_legacy:
        _run_legacy_rotation()

    if ns.compact:
        if not ns.symbol or not ns.interval:
            _print("❌ --compact kræver både --symbol og --interval.")
            return 2
        date_str = ns.date or _today_utc_str()
        return _run_compact(ns.symbol, ns.interval, date_str, ns.dry_run)

    _print("✅ Rotation complete (ingen kompaktering kørt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
