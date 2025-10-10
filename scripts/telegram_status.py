from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from utils.project_path import PROJECT_ROOT
from utils.report_utils import build_telegram_summary

# bot/telegram_status.py


TOKEN = "DIN_BOT_TOKEN"


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = build_telegram_summary(
        run_id="PROD",
        # AUTO PATH CONVERTED
        portfolio_metrics_path=PROJECT_ROOT
        / "outputs"
        / "portfolio_metrics_latest.csv",
    )
    await update.message.reply_text(msg, parse_mode="HTML")


if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("status", status_command))
    app.run_polling()
