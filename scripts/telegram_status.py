# bot/telegram_status.py

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from utils.report_utils import build_telegram_summary

TOKEN = "DIN_BOT_TOKEN"

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = build_telegram_summary(
        run_id="PROD", 
        portfolio_metrics_path="outputs/portfolio_metrics_latest.csv"
    )
    await update.message.reply_text(msg, parse_mode='HTML')

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("status", status_command))
    app.run_polling()
