# utils/logs_utils.py
"""
CSV-vedligehold til AI trading botten – robust og skalerbar.

Funktioner:
- safe_read_csv(path, **pd_kwargs) -> DataFrame
- rotate_csv(path, keep_last_rows=50_000) -> bool
- rotate_csv_streaming(path, keep_last_rows=50_000, chunk_rows=200_000) -> bool
- CLI med fil-lister, globs og mappe-scan:
    python -m utils.logs_utils --file logs/fills.csv --keep 50000
    python -m utils.logs_utils --file logs/fills.csv logs/signals.csv --keep 50000
    python -m utils.logs_utils --glob "logs/*.csv" --keep 50000
    python -m utils.logs_utils --dir logs --recursive --keep 50000
Valg:
  --dry-run       Vis hvad der ville blive trimmed, uden at skrive
  --verbose       Print ekstra status
  --size-threshold-mb  Skift automatisk til streaming over denne størrelse (default 256)
  --chunk-rows    Rækker pr. chunk ved streaming (default 200k)

Bemærk:
- Bevarer header og skriver atomisk (tempfil -> move).
- Tolererer “snavsede” rækker (on_bad_lines='skip', low_memory=False).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# -----------------------------
# Sikker læsning
# -----------------------------
def safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Sikker CSV-læsning med retrier og tolerante defaults.
    Ekstra kwargs sendes videre til pandas.read_csv (fx parse_dates=['ts']).
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    kwargs.setdefault("on_bad_lines", "skip")
    kwargs.setdefault("low_memory", False)

    last_err: Optional[Exception] = None
    for _ in range(3):  # simple retry mod fil-locks (Windows)
        try:
            return pd.read_csv(p, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise RuntimeError(f"Kunne ikke læse CSV: {p} ({last_err})")


# -----------------------------
# Rotation (in-memory)
# -----------------------------
def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), str(target))


def rotate_csv(path: str | Path, keep_last_rows: int = 50_000) -> bool:
    """
    Trim CSV til de sidste N rækker. Returnerer True hvis filen blev ændret.
    Bruges til små/mellemstore filer (hurtigt og simpelt).
    """
    p = Path(path)
    if not p.exists():
        return False
    df = pd.read_csv(p, on_bad_lines="skip", low_memory=False)
    if len(df) <= keep_last_rows:
        return False
    tail = df.tail(keep_last_rows)
    _atomic_write_csv(tail, p)
    return True


# -----------------------------
# Rotation (streaming for store filer)
# -----------------------------
def rotate_csv_streaming(
    path: str | Path,
    keep_last_rows: int = 50_000,
    *,
    chunk_rows: int = 200_000,
) -> bool:
    """
    Trim CSV til de sidste N rækker med lav RAM: læs i large chunks og behold kun halen.
    - chunk_rows: rækker pr. chunk (styrer RAM-forbrug)
    """
    p = Path(path)
    if not p.exists():
        return False

    # deque af chunks; hold samlet antal rækker ~ keep_last_rows
    frames: deque[pd.DataFrame] = deque()
    total_rows = 0
    for chunk in pd.read_csv(p, chunksize=chunk_rows, on_bad_lines="skip", low_memory=False):
        frames.append(chunk)
        total_rows += len(chunk)
        # Fjern ældste chunk(s) så vi maksimalt holder ~ keep_last_rows + 1 chunk i RAM
        while frames and (sum(len(x) for x in frames) > keep_last_rows + chunk_rows):
            frames.popleft()

    if total_rows <= keep_last_rows:
        return False

    tail_df = pd.concat(list(frames), ignore_index=True).tail(keep_last_rows)
    _atomic_write_csv(tail_df, p)
    return True


# -----------------------------
# Hjælpere til måludvælgelse
# -----------------------------
def _expand_targets(
    files: Sequence[str] | None,
    globs: Sequence[str] | None,
    dir_path: str | None,
    recursive: bool,
) -> List[Path]:
    targets: List[Path] = []

    if files:
        for f in files:
            p = Path(f)
            if p.exists():
                targets.append(p)

    if globs:
        for pattern in globs:
            base = Path(pattern).parent if any(ch in pattern for ch in ["*", "?", "["]) else Path(".")
            base = base if str(base) not in ("", ".") else Path(".")
            for candidate in base.rglob("*" if recursive else "*"):
                if candidate.is_file() and fnmatch.fnmatch(str(candidate), pattern):
                    targets.append(candidate)

    if dir_path:
        base = Path(dir_path)
        it = base.rglob("*.csv") if recursive else base.glob("*.csv")
        targets.extend([p for p in it if p.is_file()])

    # dedup + sort
    out = sorted(set(p.resolve() for p in targets))
    return out


def _file_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


# -----------------------------
# CLI
# -----------------------------
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Trim CSV-logs til de sidste N rækker (bevar header, atomisk skrivning).")

    ap.add_argument("--file", nargs="+", help="Én eller flere CSV-filer (fx logs/fills.csv logs/signals.csv)")
    ap.add_argument("--glob", nargs="+", help="Globs, fx 'logs/*.csv' 'outputs/*_signals.csv'")
    ap.add_argument("--dir", help="Mappe at scanne for *.csv")
    ap.add_argument("--recursive", action="store_true", help="Scan mapper rekursivt (med --dir eller --glob)")
    ap.add_argument("--keep", type=int, default=50_000, help="Behold sidste N rækker (default 50k)")

    ap.add_argument("--dry-run", action="store_true", help="Vis kun hvad der ville blive trimmed")
    ap.add_argument("--verbose", action="store_true", help="Udskriv ekstra status")
    ap.add_argument("--size-threshold-mb", type=float, default=256.0, help="Skift til streaming over denne størrelse (MB)")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="Rækker pr. chunk ved streaming")

    args = ap.parse_args()

    targets = _expand_targets(args.file, args.glob, args.dir, args.recursive)
    if not targets:
        print("[INFO] Ingen mål fundet. Angiv --file, --glob eller --dir.")
        return

    trimmed, skipped, failed = 0, 0, 0
    for p in targets:
        try:
            size_mb = _file_size_mb(p)
            if args.verbose:
                print(f"[...] {p} ({size_mb:.1f} MB) – keep={args.keep}")

            # Vælg strategi
            use_stream = size_mb >= float(args.size_threshold_mb)

            if args.dry_run:
                print(f"[DRY-RUN] {p} -> {'streaming' if use_stream else 'pandas.tail'}")
                continue

            if use_stream:
                changed = rotate_csv_streaming(p, keep_last_rows=args.keep, chunk_rows=args.chunk_rows)
            else:
                changed = rotate_csv(p, keep_last_rows=args.keep)

            if changed:
                print(f"[OK] Trimmet {p} → sidste {args.keep} rækker ({'stream' if use_stream else 'tail'})")
                trimmed += 1
            else:
                print(f"[SKIP] {p} var allerede ≤ {args.keep} rækker")
                skipped += 1

        except Exception as e:
            print(f"[FEJL] {p}: {e}")
            failed += 1

    print(f"Færdig: trimmed={trimmed}, skip={skipped}, fejl={failed}, total={len(targets)}")


if __name__ == "__main__":
    _cli()
