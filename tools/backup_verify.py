# -*- coding: utf-8 -*-
"""
CLI: Verificér alle backups og log status.

Brug:
  python -m bot.tools.backup_verify --backups /path/to/backups --botstatus BotStatus.md --restore /tmp/restore_dir
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from bot.utils.backup import append_botstatus, restore_backup, verify_backup


def _log_status(botstatus: Path, action: str, result: str, details: str = "") -> None:
    """Skriv én række til BotStatus.md."""
    append_botstatus(
        str(botstatus),
        {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "result": result,
            "details": details,
        },
    )


def _verify_one(
    zip_path: Path,
    botstatus: Path,
    restore_root: Path | None,
    dry_run: bool = False,
) -> Tuple[str, str]:
    """Verificér én zip og (valgfrit) restore den."""
    fn = zip_path.name
    action = f"verify:{fn}"

    try:
        ok = verify_backup(str(zip_path))
        result = "OK" if ok else "KORRUPT"
        details = "" if ok else "Checksum/manifest fejl"
    except Exception as e:  # defensiv: log utils-eksplosioner
        result = "FEJL"
        details = f"Exception: {e!r}"

    _log_status(botstatus, action, result, details)

    # Test-restore kun hvis verifikation er OK
    if result == "OK" and restore_root is not None:
        dest = restore_root / f"restore_{fn[:-4]}"
        if dry_run:
            _log_status(botstatus, f"restore:{fn}", "DRY-RUN", f"-> {dest}")
        else:
            os.makedirs(dest, exist_ok=True)
            try:
                restore_backup(str(zip_path), str(dest))
                _log_status(botstatus, f"restore:{fn}", "OK", f"-> {dest}")
            except Exception as e:
                _log_status(botstatus, f"restore:{fn}", "FEJL", str(e))
                result = "FEJL"

    return fn, result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verificér/restore backups og log til BotStatus.md"
    )
    ap.add_argument("--backups", required=True, help="Mappe med backup_*.zip")
    ap.add_argument("--botstatus", required=True, help="Sti til BotStatus.md")
    ap.add_argument("--restore", default=None, help="(Valgfri) mappe til test-restore")
    ap.add_argument("--pattern", default="*.zip", help="Glob pattern (default: *.zip)")
    ap.add_argument("--dry-run", action="store_true", help="Log uden faktisk restore")
    ap.add_argument(
        "--fail-fast", action="store_true", help="Stop ved første KORRUPT/FEJL"
    )
    args = ap.parse_args(argv)

    backups_dir = Path(args.backups).expanduser().resolve()
    botstatus = Path(args.botstatus).expanduser().resolve()
    restore_root = Path(args.restore).expanduser().resolve() if args.restore else None

    if not backups_dir.is_dir():
        print(f"[backup_verify] Mangler backup-mappe: {backups_dir}", file=sys.stderr)
        return 1
    if restore_root:
        os.makedirs(restore_root, exist_ok=True)

    files = sorted(backups_dir.glob(args.pattern))
    if not files:
        print(
            f"[backup_verify] Ingen filer matcher {args.pattern} i {backups_dir}",
            file=sys.stderr,
        )

    total = ok = corrupt = failed = 0
    for zp in files:
        if zp.suffix.lower() != ".zip":
            continue
        total += 1
        _, res = _verify_one(zp, botstatus, restore_root, args.dry_run)
        if res == "OK":
            ok += 1
        elif res == "KORRUPT":
            corrupt += 1
        else:
            failed += 1
        if args.fail_fast and res != "OK":
            break

    print(f"[backup_verify] Total:{total} OK:{ok} KORRUPT:{corrupt} FEJL:{failed}")
    # Exit-koder: 0=alt OK, 2=korruption/fejl fundet, 1=bruger-/miljøfejl (returneret tidligere)
    return 0 if corrupt == 0 and failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
