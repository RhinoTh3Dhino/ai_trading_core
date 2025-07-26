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
    trade_journal_path=None
):
    if not os.path.exists(portfolio_metrics_path):
        print(f"[WARN] Filen {portfolio_metrics_path} findes ikke, springer BotStatus.md over.")
        return
    with open(portfolio_metrics_path, "r", encoding="utf-8") as f:
        metrics_csv = f.read()
    md_content = f"""# === BotStatus.md ===\n\n**Seneste kørsel:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**Version:** {version}\n**Run ID:** {run_id}\n\n## Portefølje-metrics\n\n```csv\n{metrics_csv}\n```\n{notes}\n"""
    if plot_path and os.path.exists(plot_path):
        md_content += f"\n\n![Equity Curve]({plot_path})\n"
    if trade_journal_path and os.path.exists(trade_journal_path):
        md_content += f"\n[Se detaljeret trade journal]({trade_journal_path})\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] BotStatus.md opdateret: {md_path}")

def log_to_changelog(run_id, version, notes, changelog_path="CHANGELOG.md"):
    entry = f"\n### {datetime.now().strftime('%Y-%m-%d %H:%M')} - v{version} - {run_id}\n- {notes}\n"
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[INFO] CHANGELOG.md opdateret: {changelog_path}")

# AUTO PATH CONVERTED
def print_status(portfolio_metrics_path=PROJECT_ROOT / "outputs" / "portfolio_metrics_latest.csv"):
    if not os.path.exists(portfolio_metrics_path):
        print("Ingen portefølje-metrics fundet.")
        return
    df = pd.read_csv(portfolio_metrics_path)
    print("=== Portfolio Status ===")
    print(df.to_markdown(index=False))

def build_telegram_summary(run_id, portfolio_metrics_path, version="v1.0.0", extra_msg=None):
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
    if os.path.exists(file_path):
        os.makedirs(backup_dir, exist_ok=True)
        fname = os.path.basename(file_path)
        t = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = os.path.join(backup_dir, f"{fname}.{t}.bak")
        with open(file_path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())
        print(f"[INFO] Backup oprettet: {backup_path}")

def export_trade_journal(trades_df, output_path):
    trades_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[INFO] Trade journal eksporteret: {output_path}")

# AUTO PATH CONVERTED
def log_performance_to_history(portfolio_metrics_path, history_path=PROJECT_ROOT / "outputs" / "performance_history.csv"):
    """
    Logger altid til performance_history.csv – opretter fil med default-header hvis der ikke er data.
    """
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # Hvis der ikke findes portfolio_metrics_path, lav tom default
    if not os.path.exists(portfolio_metrics_path):
        print(f"[WARN] {portfolio_metrics_path} ikke fundet – skriver dummy-række til performance_history.csv")
        cols = ["timestamp", "Navn", "Balance"]
        df = pd.DataFrame([{
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Navn": "Ingen data",
            "Balance": 0
        }])
    else:
        df = pd.read_csv(portfolio_metrics_path)
        # Sikrer at der er timestamp-kolonne
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Hvis df er tom eller mangler relevante kolonner
        if df.empty or not all(c in df.columns for c in ["Navn", "Balance"]):
            df = pd.DataFrame([{
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Navn": "Ingen data",
                "Balance": 0
            }])

    # Tilføj til historik, eller opret ny fil med header
    if os.path.exists(history_path):
        df_hist = pd.read_csv(history_path)
        df = pd.concat([df_hist, df], ignore_index=True)
    df.to_csv(history_path, index=False)
    print(f"[INFO] performance_history.csv opdateret ({len(df)} rækker).")
