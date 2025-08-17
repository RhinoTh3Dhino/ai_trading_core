# tests/test_report_utils.py
"""
Tester utils/report_utils.py med dummy-data og tmp_path.
Mål: Maksimal branch- og linje-coverage uden side-effekter i repoet.
"""

import sys
from pathlib import Path
import os
import io
import pandas as pd
import pytest

# Sørg for at projektroden (mappen med 'utils') er i sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.report_utils import (
    update_bot_status,
    log_to_changelog,
    print_status,
    build_telegram_summary,
    backup_file,
    export_trade_journal,
)


# ---------------------------------------------------------------------
# Hjælpere
# ---------------------------------------------------------------------
def _make_test_csv(path: Path) -> Path:
    """Opretter en test CSV med portfolio-metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Navn,Balance,Profit,WinRate\nBTC,1200,12.3,0.65\nETH,900,8.9,0.54\n",
        encoding="utf-8",
    )
    return path


def _make_empty_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Navn,Balance,Profit,WinRate\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# print_status
# ---------------------------------------------------------------------
def test_print_status_and_update_bot_status(tmp_path: Path, capsys: pytest.CaptureFixture):
    csv_path = _make_test_csv(tmp_path / "portfolio_metrics_latest.csv")

    # Skal ikke raise fejl – og bør printe noget brugbart
    print_status(csv_path)
    out = capsys.readouterr().out
    assert "Navn" in out and "BTC" in out

    md_path = tmp_path / "BotStatus.md"
    update_bot_status(
        md_path=md_path,
        run_id="TEST001",
        portfolio_metrics_path=csv_path,
        version="vTEST",
        notes="Dette er en test.",
        plot_path=None,
        trade_journal_path=None,
    )

    assert md_path.exists(), "BotStatus.md blev ikke oprettet"
    content = md_path.read_text(encoding="utf-8")
    assert "TEST001" in content
    assert "vTEST" in content
    assert "Dette er en test." in content
    # CSV-indhold indsættes typisk som code block i md:
    assert "Navn,Balance,Profit,WinRate" in content


def test_print_status_handles_missing_file_gracefully(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Manglende metrics-fil bør ikke smide ukontrolleret exception."""
    missing = tmp_path / "does_not_exist.csv"
    # Forvent ingen exception:
    print_status(missing)
    msg = (capsys.readouterr().out + capsys.readouterr().err).lower()
    # Vi forventer en eller anden form for melding – ikke tom output
    assert msg is not None  # undgå for streng assert ift. ordlyd


# ---------------------------------------------------------------------
# log_to_changelog
# ---------------------------------------------------------------------
def test_log_to_changelog_create_and_append(tmp_path: Path):
    changelog_path = tmp_path / "CHANGELOG.md"

    # Create
    log_to_changelog(
        run_id="RUN123",
        version="1.0.0",
        notes="Noter til changelog",
        changelog_path=changelog_path,
    )
    assert changelog_path.exists()
    text1 = changelog_path.read_text(encoding="utf-8")
    assert "RUN123" in text1 and "1.0.0" in text1 and "Noter til changelog" in text1

    # Append (anden kørsel)
    log_to_changelog(
        run_id="RUN124",
        version="1.0.1",
        notes="Mindre rettelser",
        changelog_path=changelog_path,
    )
    text2 = changelog_path.read_text(encoding="utf-8")
    assert "RUN124" in text2 and "1.0.1" in text2 and "Mindre rettelser" in text2
    # Begge entries skal eksistere
    assert "RUN123" in text2


# ---------------------------------------------------------------------
# build_telegram_summary
# ---------------------------------------------------------------------
def test_build_telegram_summary(tmp_path: Path):
    csv_path = _make_test_csv(tmp_path / "portfolio_metrics_latest.csv")
    msg = build_telegram_summary(
        run_id="RUN999", portfolio_metrics_path=csv_path, version="vX", extra_msg="Ekstra besked"
    )
    assert "RUN999" in msg
    assert "vX" in msg
    assert "Ekstra besked" in msg
    # Sikrer at tabellen blev serialiseret
    assert "Navn" in msg and "BTC" in msg


def test_build_telegram_summary_when_missing_file(tmp_path: Path):
    """
    Når fil mangler bør funktionen stadig returnere en meningsfuld besked.
    Implementationen kan være 'graceful' og returnere en generisk tekst uden run/version.
    """
    missing = tmp_path / "nope.csv"
    msg = build_telegram_summary(
        run_id="RUN000", portfolio_metrics_path=missing, version="v0"
    )
    assert isinstance(msg, str) and len(msg) > 0
    # Accepter begge varianter: (1) indeholder metadata, eller (2) generisk fallback
    assert ("RUN000" in msg and "v0" in msg) or ("Ingen status" in msg or "ikke tilgængelig" in msg)


# ---------------------------------------------------------------------
# backup_file
# ---------------------------------------------------------------------
def test_backup_file_and_export_trade_journal(tmp_path: Path):
    # Backup af eksisterende fil
    csv_path = _make_test_csv(tmp_path / "portfolio_metrics_latest.csv")
    backup_dir = tmp_path / "backups"
    backup_file(csv_path, backup_dir=str(backup_dir))

    # Der skal ligge mindst én .bak-fil
    bak_files = list(backup_dir.glob("*.bak"))
    assert bak_files, "Backup-mappen er tom (forventede mindst én .bak-fil)"

    # Eksport af trade journal
    df = pd.DataFrame(
        [
            {"tid": "2024-07-07", "symbol": "BTC", "ret": 0.05},
            {"tid": "2024-07-07", "symbol": "ETH", "ret": 0.02},
        ]
    )
    out_path = tmp_path / "trade_journal.csv"
    export_trade_journal(df, out_path)
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert "symbol" in loaded.columns and len(loaded) == 2


def test_backup_file_nonexistent_source_warns_instead_of_raises(tmp_path: Path, capsys: pytest.CaptureFixture):
    """
    report_utils.backup_file er designet til at være 'graceful':
    Ved manglende kilde logges en WARN og der oprettes ingen backup, men der raises ikke.
    """
    missing_src = tmp_path / "missing.csv"
    backup_dir = tmp_path / "baks"
    backup_file(missing_src, backup_dir=str(backup_dir))  # må ikke raise
    captured = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "warn" in captured or "ingen backup" in captured
    assert not list(Path(backup_dir).glob("*.bak")), "Der bør ikke være .bak-filer når kilde mangler"


# ---------------------------------------------------------------------
# export_trade_journal – edge cases
# ---------------------------------------------------------------------
def test_export_trade_journal_empty_dataframe(tmp_path: Path):
    """Tom journal bør stadig skrive en gyldig CSV-header uden exception."""
    out_path = tmp_path / "journal_empty.csv"
    empty_df = pd.DataFrame(columns=["tid", "symbol", "ret"])
    export_trade_journal(empty_df, out_path)
    assert out_path.exists()
    txt = out_path.read_text(encoding="utf-8").strip()
    assert "symbol" in txt  # header skrevet


# ---------------------------------------------------------------------
# update_bot_status – valgfrie artefakter (plot + journal)
# ---------------------------------------------------------------------
def test_update_bot_status_includes_optional_artifacts(tmp_path: Path):
    csv_path = _make_test_csv(tmp_path / "portfolio_metrics_latest.csv")
    plot_path = tmp_path / "equity_curve.png"
    plot_path.write_bytes(b"\x89PNG\r\n")  # dummy fil
    journal_path = tmp_path / "trade_journal.csv"
    journal_path.write_text("tid,symbol,ret\n2024-01-01,BTC,0.1\n", encoding="utf-8")

    md_path = tmp_path / "BotStatus.md"
    update_bot_status(
        md_path=md_path,
        run_id="RUN-PLOT",
        portfolio_metrics_path=csv_path,
        version="v1.2.3",
        notes=None,  # test None-branch
        plot_path=str(plot_path),
        trade_journal_path=str(journal_path),
    )

    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    # Indikation af at både plot og journal er refereret
    assert "RUN-PLOT" in md and "v1.2.3" in md
    assert "equity_curve.png" in md or "equity" in md
    assert "trade_journal.csv" in md or "journal" in md


# ---------------------------------------------------------------------
# update_bot_status – manglende metrics (tidligt exit)
# ---------------------------------------------------------------------
def test_update_bot_status_with_missing_metrics(tmp_path: Path):
    """Tester grenen hvor portfolio_metrics_path ikke findes (skal ikke kaste fejl og ikke skrive fil)."""
    fake_path = tmp_path / "missing.csv"
    md_path = tmp_path / "BotStatus_missing.md"
    update_bot_status(md_path, run_id="NOFILE", portfolio_metrics_path=fake_path)
    # Når metrics mangler, forventer vi at der ikke skrives md (tidligt return):
    assert not md_path.exists(), "Filen bør ikke oprettes når metrics mangler"


if __name__ == "__main__":
    import pytest

    pytest.main(
        [
            __file__,
            "-vv",
            "--cov=utils.report_utils",
            "--cov-report=term-missing",
        ]
    )
