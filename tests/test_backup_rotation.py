# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path

from bot.utils.backup import create_backup, rotate_backups, verify_backup


def _write_demo_files(src: Path) -> None:
    (src / "a.txt").write_text("A", encoding="utf-8")
    (src / "b.txt").write_text("B", encoding="utf-8")


def _create_backup_unique(src: Path, bkp: Path) -> str:
    """
    Opretter en backup og sikrer, at antal .zip i mappen stiger med 1.
    Hvis tidsstempel kolliderer (samme sekund), sover vi og prøver igen.
    """
    before = {f for f in os.listdir(bkp) if f.endswith(".zip")}
    zp = create_backup(str(src), str(bkp))
    after = {f for f in os.listdir(bkp) if f.endswith(".zip")}

    if len(after) == len(before):
        # sandsynligvis navnekollision pga. samme sekund → prøv igen efter 1.1s
        time.sleep(1.1)
        zp = create_backup(str(src), str(bkp))
        after = {f for f in os.listdir(bkp) if f.endswith(".zip")}

    assert len(after) == len(before) + 1, "Backup blev ikke oprettet unikt"
    return zp


def test_backup_create_rotate(tmp_path):
    src = tmp_path / "src"
    bkp = tmp_path / "backups"
    src.mkdir()
    bkp.mkdir()
    _write_demo_files(src)

    # Opret 4 unikke backups (robust imod sekund-opløst timestamp)
    for _ in range(4):
        _create_backup_unique(src, bkp)

    zips = [f for f in os.listdir(bkp) if f.endswith(".zip")]
    assert len(zips) == 4

    deleted = rotate_backups(str(bkp), keep_last=2)
    assert deleted == 2
    zips_after = [f for f in os.listdir(bkp) if f.endswith(".zip")]
    assert len(zips_after) == 2

    # Verificér de to tilbageværende backups
    for fn in zips_after:
        assert verify_backup(os.path.join(bkp, fn)) is True
