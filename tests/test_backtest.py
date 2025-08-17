"""
tests/test_backup.py

Tester backup-funktionalitet i utils/backup.py med dummy-filer.
"""

import sys
from pathlib import Path

# Sørg for at projektroden er i sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
from utils import backup


def test_make_backup_creates_folder(tmp_path):
    # Opret en dummy-fil som skal tages backup af
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("testdata")

    backup_dir = tmp_path / "backups"
    backup_path = backup.make_backup(
        backup_folders=[str(dummy_file)],
        backup_dir=str(backup_dir),
        keep_days=1,
        keep_per_day=1
    )

    assert backup_dir.exists(), "Backup-mappen blev ikke oprettet"
    assert any(backup_dir.rglob("*")), "Ingen backup-filer fundet"
    assert os.path.exists(backup_path), "Backup-stien findes ikke"


def test_cleanup_removes_old_days_and_limits_per_day(tmp_path):
    # Opret gamle backup-mapper
    old_day_folder = tmp_path / "2020-01-01"
    old_backup_folder = old_day_folder / "backup_00-00-00"
    old_backup_folder.mkdir(parents=True)
    (old_backup_folder / "file.txt").write_text("old")

    # Opret ny backup-mappe
    new_day_folder = tmp_path / time.strftime("%Y-%m-%d")
    new_backup_folder = new_day_folder / "backup_00-00-01"
    new_backup_folder.mkdir(parents=True)
    (new_backup_folder / "file.txt").write_text("new")

    # Kør oprydning – behold kun nyere mapper
    backup.cleanup_old_backups(
        str(tmp_path), keep_days=0, keep_per_day=1
    )

    # Den gamle mappe skal være væk
    assert not old_day_folder.exists(), "Gammel backup-mappe blev ikke slettet"
    # Den nye skal stadig eksistere
    assert new_day_folder.exists(), "Ny backup-mappe blev fejlagtigt slettet"


def test_make_backup_with_nonexistent_file(tmp_path):
    # Forsøg at tage backup af en ikke-eksisterende fil
    backup_dir = tmp_path / "backups"
    result = backup.make_backup(
        backup_folders=[str(tmp_path / "missing.txt")],
        backup_dir=str(backup_dir),
        keep_days=1,
        keep_per_day=1
    )
    assert result is None or os.path.exists(result), "Backup returnerede ugyldig sti"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-vv", "--cov=utils/backup.py", "--cov-report=term-missing"])
