import os
from datetime import datetime

import pandas as pd

from utils.project_path import PROJECT_ROOT


def update_bot_status(
    md_path,
    run_id,
    portfolio_metrics_path,
    version="v1.0.0",
    notes="",
    plot_path=None,
    trade_journal_path=None,
    force_dummy=False,
):
    """Opdaterer BotStatus.md med seneste run-data."""
    if not os.path.exists(portfolio_metrics_path) and not force_dummy:
        msg = f"[WARN] Filen {portfolio_metrics_path} findes ikke, springer BotStatus.md over."
        print(msg)
        return msg

    if force_dummy:
        metrics_csv = "Dummy,0"
    else:
        with open(portfolio_metrics_path, "r", encoding="utf-8") as f:
            metrics_csv = f.read()

    md_content = (
        f"# === BotStatus.md ===\n\n"
        f"**Seneste kørsel:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Version:** {version}\n"
        f"**Run ID:** {run_id}\n\n"
        f"## Portefølje-metrics\n\n```csv\n{metrics_csv}\n```\n{notes}\n"
    )

    if plot_path and os.path.exists(plot_path):
        md_content += f"\n\n![Equity Curve]({plot_path})\n"
    if trade_journal_path and os.path.exists(trade_journal_path):
        md_content += f"\n[Se detaljeret trade journal]({trade_journal_path})\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] BotStatus.md opdateret: {md_path}")
    return md_content


def log_to_changelog(run_id, version, notes, changelog_path="CHANGELOG.md"):
    """Tilføjer en entry til CHANGELOG.md."""
    entry = (
        f"\n### {datetime.now().strftime('%Y-%m-%d %H:%M')} - v{version} - {run_id}\n- {notes}\n"
    )
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[INFO] CHANGELOG.md opdateret: {changelog_path}")
    return entry


def print_status(
    portfolio_metrics_path=PROJECT_ROOT / "outputs" / "portfolio_metrics_latest.csv",
):
    """Printer portefølje-metrics som markdown."""
    if not os.path.exists(portfolio_metrics_path):
        msg = "Ingen portefølje-metrics fundet."
        print(msg)
        return msg
    df = pd.read_csv(portfolio_metrics_path)
    table = df.to_markdown(index=False)
    print("=== Portfolio Status ===")
    print(table)
    return table


def build_telegram_summary(run_id, portfolio_metrics_path, version="v1.0.0", extra_msg=None):
    """Bygger en Telegram-besked baseret på metrics."""
    if not os.path.exists(portfolio_metrics_path):
        return "Ingen status tilgængelig."
    df = pd.read_csv(portfolio_metrics_path)
    msg = f"🤖 Trading Bot Status [{datetime.now().strftime('%d-%m-%Y %H:%M')}]\n"
    msg += f"Version: {version}\nRun ID: {run_id}\n"
    msg += "\n<b>Portefølje-metrics:</b>\n"
    msg += df.to_markdown(index=False)
    if extra_msg:
        msg += f"\n{extra_msg}"
    msg += "\nSe mere i BotStatus.md og outputs/"
    return msg


def backup_file(file_path, backup_dir="backups"):
    """Laver en backup-kopi af en given fil."""
    if os.path.exists(file_path):
        os.makedirs(backup_dir, exist_ok=True)
        fname = os.path.basename(file_path)
        t = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = os.path.join(backup_dir, f"{fname}.{t}.bak")
        with open(file_path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())
        print(f"[INFO] Backup oprettet: {backup_path}")
        return backup_path
    else:
        msg = f"[WARN] Filen {file_path} findes ikke – ingen backup lavet."
        print(msg)
        return None


def export_trade_journal(trades_df, output_path):
    """Eksporterer en trade journal til CSV."""
    trades_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[INFO] Trade journal eksporteret: {output_path}")
    return output_path


def log_performance_to_history(
    portfolio_metrics_path,
    history_path=PROJECT_ROOT / "outputs" / "performance_history.csv",
    force_dummy=False,
):
    """Logger altid til performance_history.csv – opretter fil med default-header hvis der ikke er data."""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    if (not os.path.exists(portfolio_metrics_path)) or force_dummy:
        print(
            f"[WARN] {portfolio_metrics_path} ikke fundet – skriver dummy-række til performance_history.csv"
        )
        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Navn": "Ingen data",
                    "Balance": 0,
                }
            ]
        )
    else:
        df = pd.read_csv(portfolio_metrics_path)
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if df.empty or not all(c in df.columns for c in ["Navn", "Balance"]):
            df = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Navn": "Ingen data",
                        "Balance": 0,
                    }
                ]
            )

    if os.path.exists(history_path):
        df_hist = pd.read_csv(history_path)
        df = pd.concat([df_hist, df], ignore_index=True)
    df.to_csv(history_path, index=False)
    print(f"[INFO] performance_history.csv opdateret ({len(df)} rækker).")
    return len(df)


if __name__ == "__main__":
    # Hurtig selvtest for coverage
    dummy_metrics = PROJECT_ROOT / "outputs" / "portfolio_metrics_latest.csv"
    os.makedirs(dummy_metrics.parent, exist_ok=True)
    pd.DataFrame([{"Navn": "BTCUSDT", "Balance": 1000}]).to_csv(dummy_metrics, index=False)

    update_bot_status("BotStatus.md", "test-run", dummy_metrics, force_dummy=False)
    log_to_changelog("test-run", "1.0.0", "Test note")
    print_status(dummy_metrics)
    print(build_telegram_summary("test-run", dummy_metrics))
    backup_file(dummy_metrics)
    export_trade_journal(pd.DataFrame([{"Trade": 1}]), "trade_journal.csv")
    log_performance_to_history(dummy_metrics, force_dummy=False)
