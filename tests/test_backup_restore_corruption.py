# -*- coding: utf-8 -*-
import os
import zipfile
from datetime import datetime, timezone

from bot.utils.backup import append_botstatus, create_backup, verify_backup


def test_corrupt_backup_detection_and_log(tmp_path):
    # Arrange: kildemappe + en fil
    src = tmp_path / "src"
    src.mkdir()
    (src / "file.txt").write_text("hej", encoding="utf-8")

    backups = tmp_path / "backups"
    backups.mkdir()

    botstatus = tmp_path / "BotStatus.md"

    # Opret en frisk backup (skal være gyldig)
    zp = create_backup(str(src), str(backups))
    assert verify_backup(zp) is True

    # Korrumper manifest.json i zip-arkivet
    with zipfile.ZipFile(zp, "a") as zf:
        zf.writestr("manifest.json", "NOT JSON")

    # Nu skal verifikation fejle
    assert verify_backup(zp) is False

    # Log status til BotStatus.md
    append_botstatus(
        str(botstatus),
        {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "action": f"verify:{os.path.basename(zp)}",
            "result": "KORRUPT",
            "details": "Manifest kunne ikke læses",
        },
    )

    # Tjek at logfilen indeholder både header og vores KORRUPT-markering
    content = botstatus.read_text(encoding="utf-8")
    assert "# BotStatus" in content
    assert "KORRUPT" in content
    assert "verify:" in content
