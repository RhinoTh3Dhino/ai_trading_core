# utils/artifacts.py
# Fase 4 – Persistens & filhygiejne utilities
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# --- Bevar eksisterende konstanter & helpers (backwards compatible) ---
DATE_FMT = "%Y%m%d"
DT_FMT = "%Y%m%d_%H%M%S"


def _ts(dt: bool = False) -> str:
    return datetime.now().strftime(DT_FMT if dt else DATE_FMT)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def stamp_name(prefix: str, version: str, ext: str, with_time: bool = False) -> str:
    t = _ts(dt=with_time)
    return f"{prefix}_{version}_{t}.{ext.lstrip('.')}"


def write_json(obj: dict, out_dir: str | Path, prefix: str, version: str, with_time=False) -> str:
    ensure_dir(out_dir)
    fn = stamp_name(prefix, version, "json", with_time)
    p = Path(out_dir) / fn
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(p)


def write_text(
    text: str, out_dir: str | Path, prefix: str, version: str, with_time=False, ext: str = "txt"
) -> str:
    ensure_dir(out_dir)
    fn = stamp_name(prefix, version, ext, with_time)
    p = Path(out_dir) / fn
    with p.open("w", encoding="utf-8") as f:
        f.write(text)
    return str(p)


def symlink_latest(path: str | Path, latest_link: str | Path):
    """Prøv at lave en 'latest' symlink; på Windows falder vi tilbage til copy."""
    path = Path(path)
    latest_link = Path(latest_link)
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.resolve())
    except Exception:
        shutil.copy2(str(path), str(latest_link))  # Windows fallback


def rotate_dir(path: str | Path, keep: int = 30, pattern: Optional[str] = None):
    """
    Simpel fil-rotation efter mtime (behold 'keep' nyeste).
    (Bevares for bagudkompatibilitet til eksisterende scripts.)
    """
    p = Path(path)
    if not p.is_dir():
        return
    files = [x for x in p.iterdir() if x.is_file()]
    if pattern:
        rx = re.compile(pattern)
        files = [f for f in files if rx.search(f.name)]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        try:
            f.unlink()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Fase 4: Parquet + partitioner + metadata + read-guards + kompaktering
# ----------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pandas skal være installeret for utils.artifacts (pip install pandas)"
    ) from e

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyarrow skal være installeret for Parquet support (pip install pyarrow)"
    ) from e

from config.config import PERSIST  # central Fase 4-konfig


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _current_git_sha() -> str:
    """Prøv at hente GITHUB_SHA eller `git rev-parse --short HEAD`."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


# ---------- Sti-layout helpers ----------
def partition_path(symbol: str, interval: str, dt: datetime | str) -> Path:
    """
    Returnér dags-partitionssti:
    outputs/live/{symbol}/{interval}/date=YYYY-MM-DD
    """
    if isinstance(dt, str):
        # tillad både YYYY-MM-DD og YYYYMMDD
        if "-" in dt:
            date_str = dt
        else:
            date_str = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
    else:
        date_str = dt.strftime("%Y-%m-%d")

    live_root = Path(PERSIST["LAYOUT"]["LIVE_ROOT"])
    day_dir = live_root / symbol / interval / f'{PERSIST["LAYOUT"]["DAY_PREFIX"]}{date_str}'
    ensure_dir(day_dir)
    return day_dir


def _existing_part_indices(day_dir: Path) -> List[int]:
    pref = PERSIST["LAYOUT"]["PART_PREFIX"]
    ext = PERSIST["LAYOUT"]["PART_EXT"]
    indices: List[int] = []
    for f in day_dir.glob(f"{pref}*{ext}"):
        m = re.match(rf"{re.escape(pref)}(\d+){re.escape(ext)}$", f.name)
        if m:
            try:
                indices.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(indices)


def next_part_path(day_dir: Path) -> Path:
    """Find næste part-filnavn som 'part-0001.parquet'."""
    pref = PERSIST["LAYOUT"]["PART_PREFIX"]
    ext = PERSIST["LAYOUT"]["PART_EXT"]
    indices = _existing_part_indices(day_dir)
    nxt = (indices[-1] + 1) if indices else 1
    return day_dir / f"{pref}{nxt:04d}{ext}"


# ---------- Parquet IO + metadata ----------
def _merge_metadata(
    existing: Optional[Dict[bytes, bytes]], updates: Dict[str, str]
) -> Dict[bytes, bytes]:
    """Flet pyarrow key_value_metadata (bytes) med str-keys/values."""
    out: Dict[bytes, bytes] = {}
    if existing:
        out.update(existing)
    for k, v in updates.items():
        out[k.encode("utf-8")] = str(v).encode("utf-8")
    return out


def write_parquet(df: "pd.DataFrame", path: Path, meta: Dict[str, str]) -> None:
    """
    Skriv DataFrame til Parquet med key-value metadata.
    Forvent kolonne 'ts' (timestamp) til downstream dedup/sort.
    """
    ensure_dir(path.parent)
    table = pa.Table.from_pandas(df, preserve_index=False)
    new_meta = _merge_metadata(table.schema.metadata, meta)
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, str(path))


def read_metadata(path: Path) -> Dict[str, str]:
    """Læs key-value metadata fra en Parquet-fil som dict[str, str]."""
    pf = pq.ParquetFile(str(path))
    md = pf.metadata.metadata or {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in md.items()}


def read_parquet(paths: Iterable[Path]) -> "pd.DataFrame":
    """Læs flere Parquet-filer til én DataFrame (uden at bevare metadata)."""
    dfs: List["pd.DataFrame"] = []
    for p in paths:
        dfs.append(pd.read_parquet(str(p)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def enforce_read_guard(paths: Iterable[Path], expected_features_version: str) -> None:
    """
    Læs metadata for alle paths og afvis hvis features_version != forventet.
    """
    for p in paths:
        md = read_metadata(p)
        fv = md.get("features_version")
        if fv is None:
            raise ValueError(f"Read-guard: {p} mangler 'features_version' i metadata.")
        if str(fv) != str(expected_features_version):
            raise ValueError(
                f"Read-guard: {p} har features_version={fv}, forventet={expected_features_version}."
            )


# ---------- Kompaktering & dedup ----------
def _chunker(df: "pd.DataFrame", size: int) -> Iterable["pd.DataFrame"]:
    for i in range(0, len(df), size):
        yield df.iloc[i : i + size]  # noqa: E203


def compact_day(day_dir: Path, expected_features_version: str) -> Dict[str, int]:
    """
    Dedup + kompakter en dagsmappe til færre 'part-*.parquet'.
    Returnerer stats: {'in_files','out_files','in_rows','out_rows','dropped_dups'}
    """
    pref = PERSIST["LAYOUT"]["PART_PREFIX"]
    ext = PERSIST["LAYOUT"]["PART_EXT"]

    parts = sorted(day_dir.glob(f"{pref}*{ext}"))
    if not parts:
        return {"in_files": 0, "out_files": 0, "in_rows": 0, "out_rows": 0, "dropped_dups": 0}

    # Read-guard først
    enforce_read_guard(parts, expected_features_version)

    # Læs, dedup og sortér
    df = read_parquet(parts)
    in_rows = len(df)
    if "ts" not in df.columns:
        raise ValueError(f"Kompaktering kræver kolonnen 'ts'. Manglede i {day_dir}")
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    out_rows = len(df)
    dropped = in_rows - out_rows

    # Skriv i chunks (fx 50k)
    max_rows = int(PERSIST["ROTATE_MAX_ROWS"]) or 50_000
    # Slet gamle filer først, men gem fallback hvis noget fejler
    backup_dir = day_dir / "_old_parts_tmp"
    ensure_dir(backup_dir)
    for p in parts:
        p.rename(backup_dir / p.name)

    out_files = 0
    try:
        git_sha = _current_git_sha()
        base_meta = {
            "schema_version": PERSIST["SCHEMA_VERSION"],
            "features_version": PERSIST["FEATURES_VERSION"],
            "generator": "compact_day",
            "git_sha": git_sha,
            "created_utc": _utc_now_iso(),
        }
        for chunk in _chunker(df, max_rows):
            out_path = next_part_path(day_dir)
            write_parquet(chunk, out_path, meta=base_meta)
            out_files += 1
        # Slet backup når alt lykkes
        shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        # Rollback: læg gamle filer tilbage
        for p in backup_dir.glob(f"{pref}*{ext}"):
            p.rename(day_dir / p.name)
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise

    return {
        "in_files": len(parts),
        "out_files": out_files,
        "in_rows": in_rows,
        "out_rows": out_rows,
        "dropped_dups": dropped,
    }


# ---------- Rotation (skabelon) ----------
def rotate_partition(
    symbol: str,
    interval: str,
    df_buffer: "pd.DataFrame",
    date: datetime | str | None = None,
) -> Path:
    """
    Skriv buffer til næste 'part-*.parquet' i dags-partition for (symbol, interval).
    Anvender PERSIST-metadata automatisk.
    - Hvis 'date' er None, forsøger vi at udlede dato fra df_buffer['ts'] (min-UTC),
      ellers falder vi tilbage til dagens dato (UTC).
    - 'date' kan være 'YYYY-MM-DD' eller datetime.
    """
    if df_buffer is None or df_buffer.empty:
        raise ValueError("rotate_partition: df_buffer er tom.")

    # Udled dato fra data hvis ikke angivet
    if date is None and "ts" in df_buffer.columns and len(df_buffer["ts"]) > 0:
        ts0 = pd.to_datetime(df_buffer["ts"].min(), utc=True)
        date_str = ts0.strftime("%Y-%m-%d")
        day_dir = partition_path(symbol, interval, date_str)
    else:
        dt = date if date is not None else datetime.now(timezone.utc)
        day_dir = partition_path(symbol, interval, dt)

    out_path = next_part_path(day_dir)
    meta = {
        "schema_version": PERSIST["SCHEMA_VERSION"],
        "features_version": PERSIST["FEATURES_VERSION"],
        "generator": "rotate_partition",
        "git_sha": _current_git_sha(),
        "created_utc": _utc_now_iso(),
    }
    write_parquet(df_buffer, out_path, meta=meta)
    return out_path


def rotate_dir_f4(root: str | Path) -> Dict[str, int]:
    """
    Fase-4 rotation på mappe-niveau (no-op skabelon).
    Denne funktion kan udbygges til at:
      - inspicere åbne buffers
      - rotere når PERSIST-thresholds overskrides
    Returnerer simple stats for kompatibilitet.
    """
    p = Path(root)
    if not p.exists():
        return {"rotated": 0}
    # Her kunne man implementere tidsbaseret rotation ud fra mtime/markerfiler.
    # I praksis håndteres “buffer → part” typisk i live-pipelinen via rotate_partition().
    return {"rotated": 0}
