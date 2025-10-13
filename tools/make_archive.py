# tools/make_archive.py
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

from config.config import PERSIST
from utils.artifacts import ensure_dir  # genbrug helper

# Hvad skal med i arkivet (sti, required)
INCLUDE: List[Tuple[str, bool]] = [
    ("outputs/live", True),  # Ny: hele live-outputs med partitioner
    ("outputs/models", True),
    ("outputs/backtests", True),
    ("outputs/metrics", True),
    ("outputs/charts", True),
    ("outputs/feature_data", False),
    ("outputs/labels", False),
    ("docs", True),
    ("BotStatus.md", False),
    ("CHANGELOG.md", False),
]

ARCHIVE_DIR = Path("archives")
ARCHIVE_PREFIX = "trading-core-archive"


def _utc_stamp() -> str:
    # YYYYMMDD_HHMMSSZ
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _git_sha_short() -> str:
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


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def _collect_sources(tmpdir: Path) -> List[dict]:
    """Kopier kilder til tmpdir og returnér metadata for manifest."""
    copied: List[dict] = []
    for raw, required in INCLUDE:
        src = Path(raw)
        if not src.exists():
            msg = f"⚠ mangler: {src}"
            if required:
                print(msg)
            else:
                print(f"(skipper) {msg}")
            continue

        dest = tmpdir / src
        ensure_dir(dest.parent)

        try:
            if src.is_dir():
                # copytree i py>=3.8 understøtter dirs_exist_ok
                shutil.copytree(src, dest, dirs_exist_ok=True)
                size = _dir_size_bytes(src)
                kind = "dir"
            else:
                shutil.copy2(src, dest)
                size = src.stat().st_size
                kind = "file"
            copied.append(
                {"path": str(src), "required": required, "kind": kind, "size_bytes": int(size)}
            )
            print(f"✓ kopieret: {src} -> {dest}")
        except Exception as e:
            print(f"❌ fejl ved kopiering af {src}: {e}")
            if required:
                raise
    return copied


def _write_manifest(tmpdir: Path, entries: List[dict]) -> None:
    meta = {
        "created_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "git_sha": _git_sha_short(),
        "schema_version": PERSIST["SCHEMA_VERSION"],
        "features_version": PERSIST["FEATURES_VERSION"],
        "entries": entries,
    }
    mf = tmpdir / "manifest.json"
    mf.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ℹ manifest: {mf}")


def _prune_old_archives() -> int:
    """Slet ZIP-arkiver ældre end RETENTION_DAYS. Returnerer antal slettet."""
    days = int(PERSIST.get("RETENTION_DAYS", 30))
    if days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    deleted = 0
    for z in ARCHIVE_DIR.glob(f"{ARCHIVE_PREFIX}_*.zip"):
        try:
            mtime = datetime.fromtimestamp(z.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                z.unlink(missing_ok=True)
                deleted += 1
        except Exception:
            pass
    if deleted:
        print(f"🧹 pruned {deleted} gamle arkiver (> {days} dage)")
    return deleted


def make_archive() -> Path:
    ensure_dir(ARCHIVE_DIR)

    stamp = _utc_stamp()
    rootname = f"{ARCHIVE_PREFIX}_{stamp}"
    tmpdir = ARCHIVE_DIR / rootname
    ensure_dir(tmpdir)

    try:
        entries = _collect_sources(tmpdir)
        _write_manifest(tmpdir, entries)

        base_name = ARCHIVE_DIR / rootname
        zip_path = Path(shutil.make_archive(str(base_name), "zip", tmpdir))
        print(f"✅ Arkiv bygget: {zip_path.name} ({zip_path.stat().st_size} bytes)")

        return zip_path
    finally:
        # Oprydning uanset succes/fejl
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    try:
        zip_path = make_archive()
        _prune_old_archives()
        print(f"🎯 Færdig: {zip_path}")
        return 0
    except Exception as e:
        print(f"❌ make_archive fejlede: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
