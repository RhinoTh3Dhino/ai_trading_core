# tests/persistence/test_archive_and_retention.py
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

import tools.make_archive as MA
from utils.artifacts import partition_path, write_parquet


def _touch_old_zip(path: Path, days_old: int):
    # Sæt mtime "days_old" dage tilbage
    old = time.time() - days_old * 24 * 3600
    os.utime(path, (old, old))


def test_make_archive_and_retention(monkeypatch, tmp_path, tmp_layout):
    """
    Byg et arkiv i tmp-arbejdsdir og verificér at
    - zip bliver oprettet
    - gamle zip-arkiver prunes iht. RETENTION_DAYS
    """
    # Kør alle operationer i tmp_path som CWD
    monkeypatch.chdir(tmp_path)

    # Opret minimal struktur som make_archive forventer
    live_day = partition_path("BTCUSDT", "1m", "2025-01-02")
    df = pd.DataFrame({"ts": pd.to_datetime([1], unit="ms", utc=True), "open": [1.0]})
    write_parquet(
        df,
        live_day / "part-0001.parquet",
        meta={
            "schema_version": "1.0.0",
            "features_version": tmp_layout["FEATURES_VERSION"],
            "generator": "test",
        },
    )
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "Runbook.md").write_text("# Runbook", encoding="utf-8")

    # Lav et gammelt arkiv og sæt mtime 40 dage tilbage
    MA.ARCHIVE_DIR.mkdir(exist_ok=True, parents=True)
    old_zip = MA.ARCHIVE_DIR / f"{MA.ARCHIVE_PREFIX}_OLD.zip"
    old_zip.write_bytes(b"dummy")
    _touch_old_zip(old_zip, days_old=40)

    # Tving retention til 30 dage for determinisme
    import config.config as cfg

    cfg.PERSIST["RETENTION_DAYS"] = 30
    # Og sørg for, at make_archive-modulets syn på PERSIST matcher
    MA.PERSIST = cfg.PERSIST

    # Kør make_archive (kalder også _prune_old_archives())
    rc = MA.main()
    assert rc == 0

    # Verificér at nyt zip eksisterer
    zips = sorted(MA.ARCHIVE_DIR.glob(f"{MA.ARCHIVE_PREFIX}_*.zip"))
    assert len(zips) >= 1

    # Det gamle arkiv (>30 dage) bør være slettet
    assert not old_zip.exists()
