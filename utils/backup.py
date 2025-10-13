# -*- coding: utf-8 -*-
"""
Backup/restore, rotation og statuslog.

Modulet understøtter to spor:

1) ZIP + manifest (SHA-256) – brugt af tests/CI:
   - create_backup(src_dir, backups_dir) -> str
   - verify_backup(zip_path) -> bool
   - restore_backup(zip_path, dest_dir) -> None
   - rotate_backups(backups_dir, keep_last=5) -> int
   - append_botstatus(botstatus_path, entry: dict) -> None

2) Mappe-baseret dagsbackup (din oprindelige funktionalitet):
   - make_backup(backup_folders=None, backup_dir="backups", keep_days=7, keep_per_day=10, ...)
   - cleanup_old_backups(backup_dir, keep_days=7, keep_per_day=10) -> list[str]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .errors import BackupError

__all__ = [
    "create_backup",
    "verify_backup",
    "restore_backup",
    "rotate_backups",
    "append_botstatus",
    "make_backup",
    "cleanup_old_backups",
]

# ---------------------------------------------------------------------
# Utils (fælles)
# ---------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# ZIP + manifest-API (bruges i robusthedstests)
# ---------------------------------------------------------------------


def create_backup(src_dir: str | Path, backups_dir: str | Path) -> str:
    """
    Pakker hele src_dir i en zip med manifest (checksums). Returnerer sti til zip.
    """
    src = Path(src_dir)
    out_dir = Path(backups_dir)
    _ensure_dir(out_dir)

    # UTC timestamp i filnavn for deterministisk sortering
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = out_dir / f"backup_{ts}.zip"

    manifest: Dict[str, object] = {"created_utc": ts, "files": []}

    try:
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(src):
                for fn in files:
                    full = Path(root) / fn
                    rel = str(full.relative_to(src))
                    with open(full, "rb") as fh:
                        data = fh.read()
                    zf.writestr(rel, data)
                    (manifest["files"]).append({"path": rel, "sha256": _sha256_bytes(data)})
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return str(zip_path)
    except Exception as e:
        raise BackupError(f"Kunne ikke oprette backup: {e}") from e


def verify_backup(zip_path: str | Path) -> bool:
    """
    Verificerer manifest og checksums.
    Returnerer True hvis alt matcher, ellers False.
    """
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            for entry in manifest.get("files", []):
                p = entry["path"]
                expected = entry["sha256"]
                data = zf.read(p)
                if _sha256_bytes(data) != expected:
                    return False
        return True
    except Exception:
        return False


def restore_backup(zip_path: str | Path, dest_dir: str | Path) -> None:
    """
    Udpakker backup til dest_dir.
    """
    try:
        dest = Path(dest_dir)
        _ensure_dir(dest)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(dest))
    except Exception as e:
        raise BackupError(f"Kunne ikke restore backup: {e}") from e


def rotate_backups(backups_dir: str | Path, keep_last: int = 5) -> int:
    """
    Beholder de nyeste 'keep_last' ZIP-backups i 'backups_dir', sletter resten.
    Returnerer antal slettede.
    """
    bdir = Path(backups_dir)
    if not bdir.exists():
        return 0

    files = [
        f
        for f in bdir.iterdir()
        if f.is_file() and f.name.startswith("backup_") and f.suffix == ".zip"
    ]
    # Navn indeholder UTC timestamp -> sortering giver nyeste først
    files.sort(key=lambda p: p.name, reverse=True)
    to_delete = files[keep_last:]
    count = 0
    for fp in to_delete:
        try:
            fp.unlink(missing_ok=True)
            count += 1
        except Exception:
            # Ignorer individuelle sletningsfejl (kan logges andetsteds)
            pass
    return count


def append_botstatus(botstatus_path: str | Path, entry: Dict[str, str]) -> None:
    """
    Appender en række til BotStatus.md i Markdown-tabel-format.
    Sikrer tabelhoved findes.
    entry forventer nøgler: date_utc, action, result, details
    """
    botstatus = Path(botstatus_path)
    header = "| dato_utc | handling | resultat | detaljer |\n" "|---|---|---|---|\n"
    row = (
        f"| {entry.get('date_utc')} | {entry.get('action')} | "
        f"{entry.get('result')} | {entry.get('details','')} |\n"
    )

    if not botstatus.exists() or botstatus.stat().st_size == 0:
        botstatus.write_text("# BotStatus\n\n" + header + row, encoding="utf-8")
        return

    content = botstatus.read_text(encoding="utf-8")
    if "| dato_utc | handling | resultat | detaljer |" not in content:
        content += "\n" + header
    content += row
    botstatus.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------
# Mappe-baseret dagsbackup (din oprindelige funktionalitet)
# ---------------------------------------------------------------------


def make_backup(
    backup_folders: Optional[List[str]] = None,
    backup_dir: str | Path = "backups",
    keep_days: int = 7,
    keep_per_day: int = 10,
    create_dummy_if_empty: bool = True,
    force_dummy: bool = False,
) -> str | None:
    """
    Laver backup af valgte mapper/filer med timestamp i dato-undermappe og sletter gamle backups.
    Opretter dummy test-fil hvis intet andet findes, så workflow altid producerer noget.

    Returnerer stien til den oprettede backup-mappe, eller None ved fatal fejl.
    """
    if backup_folders is None:
        backup_folders = [
            "models",
            "logs",
            "tuner_cache",
            "data",
            "BotStatus.md",
            "CHANGELOG.md",
        ]

    if keep_days < 0 or keep_per_day < 0:
        raise ValueError("keep_days og keep_per_day skal være >= 0")

    backup_root = Path(backup_dir)
    print(f"📦 Forsøger at tage backup af: {backup_folders}")

    # Lav undermappe til dagens dato
    now_local = datetime.now()  # lokal tid som i din oprindelige funktion
    date_str = now_local.strftime("%Y-%m-%d")
    timestamp = now_local.strftime("%H-%M-%S")

    day_dir = backup_root / date_str
    backup_path = day_dir / f"backup_{timestamp}"

    try:
        _ensure_dir(backup_path)
        print(f"✅ Backup-mappe oprettet: {backup_path}")
    except Exception as e:
        print(f"❌ Fejl ved oprettelse af backup-mappe: {backup_path}: {e}")
        return None

    found_any = False

    for item in backup_folders:
        src = Path(item)
        print(f"🔍 Tjekker om {item} findes: {src.exists()}")
        if src.exists() and not force_dummy:
            try:
                dst = backup_path / src.name
                if src.is_dir():
                    # dirs_exist_ok for idempotens, hvis job kører flere gange samme sekund
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    # sørg for parent findes
                    _ensure_dir(dst.parent)
                    shutil.copy2(src, dst)
                print(f"✅ Backed up: {item}")
                found_any = True
            except Exception as e:
                print(f"❌ Kunne ikke backe op: {item}: {e}")
        else:
            print(f"⚠️ Advarsel: {item} findes ikke og blev ikke backet op.")

    # Dummy-fil hvis nødvendigt
    if (not found_any and create_dummy_if_empty) or force_dummy:
        try:
            dummy_path = backup_path / "dummy_backup.txt"
            dummy_path.write_text(
                "Ingen af de forventede filer/mapper fandtes - dummy backup.\n",
                encoding="utf-8",
            )
            print(f"🟡 Oprettede dummy-fil: {dummy_path}")
        except Exception as e:
            print(f"❌ Kunne ikke skrive dummy-fil: {e}")

    # Ryd op i gamle backups (ikke fatal ved fejl)
    try:
        cleanup_old_backups(backup_root, keep_days=keep_days, keep_per_day=keep_per_day)
    except Exception as e:
        print(f"❌ Fejl under sletning af gamle backups: {e}")

    return str(backup_path)


def cleanup_old_backups(
    backup_dir: str | Path,
    keep_days: int = 7,
    keep_per_day: int = 10,
) -> List[str]:
    """
    Sletter gamle backups:
    - Behold kun 'keep_days' dage (mindst 1 dag).
    - Behold maks. 'keep_per_day' backups pr. dag.
    Returnerer liste over slettede filer/mapper (stier som str).
    """
    deleted_items: List[str] = []
    root = Path(backup_dir)

    if not root.exists():
        return deleted_items

    # Altid behold mindst én dags backups
    keep_days = max(1, keep_days)

    # Ryd op i backups pr. dag
    for datedir in root.iterdir():
        if not datedir.is_dir():
            continue

        backups = [d for d in datedir.iterdir() if d.is_dir() and d.name.startswith("backup_")]
        # nyeste først (navn indeholder klokkeslæt HH-MM-SS)
        backups.sort(key=lambda p: p.name, reverse=True)

        for b in backups[keep_per_day:]:
            try:
                shutil.rmtree(b)
                deleted_items.append(str(b))
                print(f"🗑️ Slettet gammel backup: {b}")
            except Exception as e:
                print(f"❌ Kunne ikke slette: {b}: {e}")

    # Ryd op i dato-mapper, men aldrig den nyeste dag
    days = [d for d in root.iterdir() if d.is_dir()]
    days.sort(key=lambda p: p.name, reverse=True)
    for day_to_remove in days[keep_days:]:
        try:
            shutil.rmtree(day_to_remove)
            deleted_items.append(str(day_to_remove))
            print(f"🗑️ Slettet gammel backup-dag: {day_to_remove}")
        except Exception as e:
            print(f"❌ Kunne ikke slette dag: {day_to_remove}: {e}")

    return deleted_items


# ---------------------------------------------------------------------
# CLI/direkte test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        path = make_backup(force_dummy=True)
        print(f"Test-backup oprettet i: {path}")
    except Exception as e:
        print(f"Fejl ved test-backup: {e}")
