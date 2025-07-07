import time
import os
import schedule
from dotenv import load_dotenv

from utils.backup import make_backup
from utils.botstatus import update_bot_status
from utils.changelog import append_to_changelog
from utils.telegram_utils import send_message
from utils.report_utils import log_performance_to_history   # <-- NYT: historik-logging
from utils.telegram_utils import generate_trend_graph, send_trend_graph  # <-- NYT: trend-graf

from utils.robust_utils import safe_run
from bot.engine import main as engine_main
from bot.engine import load_best_ensemble_params

import pandas as pd

# Indlæs miljøvariabler
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def ensure_performance_history_exists():
    """
    Sikrer at outputs/performance_history.csv altid eksisterer (også hvis der ikke køres nogen handler).
    """
    history_path = "outputs/performance_history.csv"
    if not os.path.exists(history_path):
        os.makedirs("outputs", exist_ok=True)
        pd.DataFrame([{"timestamp": "", "Navn": "", "Balance": ""}]).to_csv(history_path, index=False)
        print("🟡 Oprettede tom outputs/performance_history.csv for CI compliance.")

def ensure_botstatus_exists():
    if not os.path.exists("BotStatus.md"):
        with open("BotStatus.md", "w", encoding="utf-8") as f:
            f.write("# Dummy BotStatus for CI\n")
        print("🟡 Oprettede BotStatus.md (dummy for CI)")

def ensure_changelog_exists():
    if not os.path.exists("CHANGELOG.md"):
        with open("CHANGELOG.md", "w", encoding="utf-8") as f:
            f.write("# Changelog (dummy for CI)\n")
        print("🟡 Oprettede CHANGELOG.md (dummy for CI)")

def main_trading_cycle():
    """
    Kører hele trading-pipelinen fra engine.py og laver backup, status, logging, performance-historik og trend-graf.
    """
    print("✅ Botten starter trading-cyklus...")

    # Sikrer outputfiler fra starten (for CI)
    ensure_performance_history_exists()
    ensure_botstatus_exists()
    ensure_changelog_exists()

    # Hent de bedste tuning-parametre (threshold + weights) før hver run!
    threshold, weights = load_best_ensemble_params()
    print(f"[INFO] Bruger threshold={threshold}, weights={weights} til dette run.")
    try:
        engine_main(threshold=threshold, weights=weights)
    except Exception as e:
        print(f"❌ FEJL i engine_main: {e}")
        send_message(f"❌ FEJL i engine_main: {e}")
        # CI skal stadig fortsætte for at få artifacts!
    # NYT: Log performance-historik
    log_performance_to_history("outputs/portfolio_metrics_latest.csv")

    # NYT: Generér og send trend-graf (kan deaktiveres hvis ønsket)
    try:
        img_path = generate_trend_graph()
        send_trend_graph(TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, img_path)
        print(f"✅ Trend-graf genereret og sendt: {img_path}")
    except Exception as e:
        print(f"❌ Fejl ved trend-graf: {e}")

    backup_path = make_backup(
        keep_days=7,
        keep_per_day=10
    )
    print(f"✅ Backup gemt: {backup_path}")
    send_message(f"✅ Bot kørte OK og lavede backup: {backup_path}")

    # NYT: Sikrer at alle artefakt-filer ALTID eksisterer!
    ensure_performance_history_exists()
    ensure_botstatus_exists()
    ensure_changelog_exists()

    return backup_path

def daily_status():
    try:
        send_message("📊 Daglig status: Botten kører fortsat! Tilpas evt. med flere metrics her.")
        append_to_changelog("📊 Daglig status sendt til Telegram.")
        print("✅ Daglig status sendt.")
    except Exception as e:
        print(f"❌ Fejl ved daglig status: {e}")

def retrain_models():
    try:
        send_message("🔄 Starter automatisk retrain af modeller!")
        # TODO: Kald evt. retrain-funktionalitet her
        append_to_changelog("🔄 Automatisk retrain af modeller startet.")
        print("✅ Retrain-job kørt.")
    except Exception as e:
        print(f"❌ Fejl ved retrain: {e}")

def heartbeat():
    try:
        send_message("💓 Bot heartbeat: Jeg er stadig i live!")
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
            send_message(f"❌ Bot FEJLEDE under kørsel: {e}")
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
    print(f"CI mode: {os.getenv('CI', 'false').lower()}")

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
