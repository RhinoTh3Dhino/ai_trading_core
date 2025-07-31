# utils/ci_notify.py

import os
import sys

# Importer Telegram-funktion (du skal have utils/telegram_utils.py i projektet)
try:
    from utils.telegram_utils import send_message
except ImportError:
    print(
        "❌ Kunne ikke importere send_message fra utils/telegram_utils.py – CI-notifikation springes over."
    )
    sys.exit(0)

# Hent CI-info fra miljøvariabler (GitHub Actions sætter disse automatisk)
ci_status = os.getenv("CI_STATUS", "KØRT")
workflow = os.getenv("GITHUB_WORKFLOW", "ukendt workflow")
run_number = os.getenv("GITHUB_RUN_NUMBER", "")
run_id = os.getenv("GITHUB_RUN_ID", "")
repo = os.getenv("GITHUB_REPOSITORY", "")
commit_sha = os.getenv("GITHUB_SHA", "")[:8]
branch = os.getenv("GITHUB_REF_NAME", "")
actor = os.getenv("GITHUB_ACTOR", "")
event = os.getenv("GITHUB_EVENT_NAME", "")
job = os.getenv("GITHUB_JOB", "")
link = f"https://github.com/{repo}/actions/runs/{run_id}"

# Sæt status/farve
if ci_status.lower() in ("success", "passed", "ok", "kørt"):
    emoji = "✅"
    status_txt = "SUCCES"
elif ci_status.lower() in ("failure", "failed", "error"):
    emoji = "❌"
    status_txt = "FEJL"
else:
    emoji = "⚠️"
    status_txt = ci_status.upper()

msg = (
    f"{emoji} <b>CI status: {status_txt}</b>\n"
    f"<b>Workflow:</b> {workflow}\n"
    f"<b>Branch:</b> {branch} ({commit_sha})\n"
    f"<b>Run:</b> #{run_number} (af {actor})\n"
    f"<b>Job:</b> {job} | Event: {event}\n"
    f"<a href='{link}'>Se run-detaljer på GitHub</a>"
)

# Send beskeden til Telegram
try:
    send_message(msg, parse_mode="HTML")
    print("✅ CI-status sendt til Telegram.")
except Exception as e:
    print(f"⚠️ Kunne ikke sende CI-status til Telegram: {e}")
    # Fail aldrig CI på grund af Telegram – kun advarsel!

if __name__ == "__main__":
    print("CI-notifikation kørt – se Telegram for status.")
