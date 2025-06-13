import time
import os
import schedule
from dotenv import load_dotenv

from utils.backup import make_backup
from utils.botstatus import update_bot_status
from utils.changelog import append_to_changelog
from utils.telegram_utils import (
    send_telegram_message,
)
from utils.robust_utils import safe_run

# Importér din engine pipeline-funktion (fra engine.py)
from bot.engine import main as engine_main

# Indlæs miljøvariabler
load_dotenv()

if os.getenv("DEBUG", "false").lower() == "true":
    print(f"DEBUG: TELEGRAM_TOKEN = {os.getenv('TELEGRAM_TOKEN')}")
    print(f"DEBUG: TELEGRAM_CHAT_ID = {os.getenv('TELEGRAM_CHAT_ID')}")

def main_trading_cycle():
    """
    Kører hele trading-pipelinen fra engine.py og laver backup, status, logging.
    """
    print("✅ Botten starter trading-cyklus...")
    engine_main()  # Kør din trading-pipeline!

    backup_path = make_backup(
        keep_days=7,
        keep_per_day=10
    )
    print(f"✅ Backup gemt: {backup_path}")
    send_telegram_message(f"✅ Bot kørte OK og lavede backup: {backup_path}")

    return backup_path

def daily_status():
    try:
        send_telegram_message("📊 Daglig status: Botten kører fortsat! Tilpas evt. med flere metrics her.")
        append_to_changelog("📊 Daglig status sendt til Telegram.")
        print("✅ Daglig status sendt.")
    except Exception as e:
        print(f"❌ Fejl ved daglig status: {e}")

def retrain_models():
    try:
        send_telegram_message("🔄 Starter automatisk retrain af modeller!")
        # TODO: Kald evt. retrain-funktionalitet her
        append_to_changelog("🔄 Automatisk retrain af modeller startet.")
        print("✅ Retrain-job kørt.")
    except Exception as e:
        print(f"❌ Fejl ved retrain: {e}")

def heartbeat():
    try:
        send_telegram_message("💓 Bot heartbeat: Jeg er stadig i live!")
        print("✅ Heartbeat sendt.")
    except Exception as e:
        print(f"❌ Fejl ved heartbeat: {e}")

def main():
    print("✅ AI Trading Bot starter...")
    error_msg = None
    backup_path = None

    try:
        backup_path = main_trading_cycle()
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Fejl under kørsel: {e}")
        try:
            send_telegram_message(f"❌ Bot FEJLEDE under kørsel: {e}")
        except Exception as tel_e:
            print(f"❌ Telegram FEJL: {tel_e}")
    finally:
        update_bot_status(
            status="✅ Succes" if error_msg is None else "❌ Fejl",
            backup_path=backup_path,
            error_msg=error_msg if error_msg else "Ingen"
        )
        if error_msg is None:
            append_to_changelog(f"✅ Bot kørte og lavede backup: {backup_path}")
        else:
            append_to_changelog(f"❌ Bot fejlede: {error_msg}")

    print("✅ Bot-kørsel færdig.")

if __name__ == "__main__":
    print("🚀 AI Trading Bot (Production Mode) starter med schedule!")

    # CI: Kør kun én cyklus og afslut!
    if os.getenv("CI", "false").lower() == "true":
        safe_run(main)
    else:
        # === Kør første trading-cyklus straks (så du ser output/Telegram med det samme) ===
        safe_run(main)

        # Kør trading-cyklus hver time
        schedule.every().hour.at(":00").do(lambda: safe_run(main))

        # Daglig status kl. 08:00
        schedule.every().day.at("08:00").do(lambda: safe_run(daily_status))

        # Retrain hver nat kl. 03:00
        schedule.every().day.at("03:00").do(lambda: safe_run(retrain_models))

        # Heartbeat hver time kl. xx:30
        schedule.every().hour.at(":30").do(lambda: safe_run(heartbeat))

        while True:
            schedule.run_pending()
            time.sleep(5)
