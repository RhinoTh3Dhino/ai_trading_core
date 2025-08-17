# tests/test_backup.py
"""
Tests for utils/backup.py

- Opretter backup i en midlertidig mappe (tmp_path)
- Tester både dummy-backup og kopi af rigtige filer/mapper
- Validerer mappeoprettelse, filindhold, navngivningskonvention og parameterfejl
- Tester cleanup af gamle dage og begrænsning pr. dag
- Tester graceful håndtering når backup-roden ikke findes
"""

import sys
from pathlib import Path
import re
import os

# Sørg for at projektroden (mappen med 'utils') er i sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from utils.backup import make_backup, cleanup_old_backups


def test_make_backup_creates_folder_and_dummy(tmp_path):
    """make_backup skal oprette dagsmappe + backup_*-mappe og en dummy-fil, når ingen kilder findes."""
    backup_dir = tmp_path / "backups"

    path_str = make_backup(
        backup_folders=["nonexistent.txt"],  # sikrer at vi rammer dummy-stien
        backup_dir=str(backup_dir),
        keep_days=1,
        keep_per_day=1,
        create_dummy_if_empty=True,
        force_dummy=True,  # stabil: altid en dummy-fil
    )
    assert path_str, "make_backup returnerede ingen sti"
    path = Path(path_str)
    assert path.exists() and path.is_dir(), f"Backup-mappe findes ikke: {path}"

    # backup-mappenavn matcher backup_*
    assert re.search(r"backup_\d{2}-\d{2}-\d{2}$", path.name), f"Uventet mappenavn: {path.name}"

    # dummy-fil skal ligge i mappen
    dummy = path / "dummy_backup.txt"
    assert dummy.exists() and dummy.is_file(), "Dummy-fil blev ikke oprettet"
    txt = dummy.read_text(encoding="utf-8").strip()
    assert txt.startswith("Ingen af de forventede") or len(txt) > 0, "Uventet dummy-indhold"


def test_make_backup_copies_real_file_and_directory(tmp_path):
    """Når gyldige kilder angives, skal de kopieres til den nye backup-mappe."""
    backup_dir = tmp_path / "backups"
    # Kilde: fil
    src_file = tmp_path / "src" / "config.yaml"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text("foo: bar\n", encoding="utf-8")

    # Kilde: mappe med indhold
    src_dir = tmp_path / "src_dir"
    (src_dir / "sub").mkdir(parents=True, exist_ok=True)
    (src_dir / "sub" / "data.txt").write_text("hello", encoding="utf-8")

    path_str = make_backup(
        backup_folders=[str(src_file), str(src_dir)],
        backup_dir=str(backup_dir),
        keep_days=7,
        keep_per_day=10,
        create_dummy_if_empty=True,  # må ikke bruges her, da vi har kilder
        force_dummy=False,
    )
    dest = Path(path_str)
    assert dest.exists() and dest.is_dir()

    # Filen skal være kopieret med filnavn
    copied_file = dest / "config.yaml"
    assert copied_file.exists()
    assert copied_file.read_text(encoding="utf-8") == "foo: bar\n"

    # Mappen skal være kopieret rekursivt
    copied_nested = dest / "src_dir" / "sub" / "data.txt"
    # Nogle implementationer bevarer mappenavnet, andre kopierer indhold — tjek begge muligheder
    if not copied_nested.exists():
        copied_nested = dest / "sub" / "data.txt"
    assert copied_nested.exists(), "Mappeindhold blev ikke kopieret rekursivt"
    assert copied_nested.read_text(encoding="utf-8") == "hello"


def test_cleanup_old_backups_limits_per_day_and_days(tmp_path):
    """cleanup_old_backups skal begrænse antal backups per dag og antal dage."""
    backup_root = tmp_path / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)

    # Opret 2 datoer: gammel (beholdes ikke) og ny (beholdes)
    old_day = backup_root / "2020-01-01"
    new_day = backup_root / "2099-01-01"
    for d in (old_day, new_day):
        d.mkdir(parents=True, exist_ok=True)
        # Opret 3 backups for at teste keep_per_day=1
        for i in range(3):
            b = d / f"backup_00-00-{i:02d}"
            b.mkdir(parents=True, exist_ok=True)
            (b / "file.txt").write_text("x", encoding="utf-8")

    deleted = cleanup_old_backups(str(backup_root), keep_days=1, keep_per_day=1)

    # Den gamle dag skal slettes
    assert not old_day.exists(), "Gammel dagsmappe blev ikke slettet"
    # Den nye dag skal blive, men kun 1 backup-mappe må være tilbage
    assert new_day.exists(), "Ny dagsmappe blev fejlagtigt slettet"
    remaining = [p for p in new_day.iterdir() if p.is_dir() and p.name.startswith("backup_")]
    assert len(remaining) == 1, f"For mange backups tilbage for ny dag: {remaining}"
    assert isinstance(deleted, list), "cleanup_old_backups bør returnere en liste over slettede stier"


def test_cleanup_old_backups_handles_missing_root(tmp_path):
    """Når backup-roden ikke findes, bør funktionen ikke fejle og returnere en tom liste."""
    missing_root = tmp_path / "no_backups_here"
    # Må ikke raise
    deleted = cleanup_old_backups(str(missing_root), keep_days=7, keep_per_day=3)
    assert isinstance(deleted, list)
    assert deleted == []


def test_make_backup_invalid_parameters(tmp_path):
    """Negativ keep_days/keep_per_day skal give ValueError."""
    backup_dir = tmp_path / "backups"
    with pytest.raises(ValueError):
        make_backup(backup_dir=str(backup_dir), keep_days=-1)
    with pytest.raises(ValueError):
        make_backup(backup_dir=str(backup_dir), keep_per_day=-5)
