import time

import schedule
from telegram import Bot

from utils.project_path import PROJECT_ROOT
from utils.report_utils import build_telegram_summary

# scripts/auto_status.py


TELEGRAM_TOKEN = "DIN_BOT_TOKEN"
CHAT_ID = "DIT_CHAT_ID"


def send_daily_status():
    msg = build_telegram_summary(
        run_id="AUTO",
        # AUTO PATH CONVERTED
        portfolio_metrics_path=PROJECT_ROOT / "outputs" / "portfolio_metrics_latest.csv",
    )
    bot = Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")


# Kør hver dag kl 08:00
schedule.every().day.at("08:00").do(send_daily_status)

if __name__ == "__main__":
    print("Scheduler started. Tryk Ctrl+C for at stoppe.")
    while True:
        schedule.run_pending()
        time.sleep(30)
