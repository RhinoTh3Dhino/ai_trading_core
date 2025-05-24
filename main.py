import time
import os
from utils.backup import make_backup
from utils.botstatus import update_bot_status
from utils.changelog import append_to_changelog
from utils.telegram_utils import send_telegram_message  # ← Tilføjet import
from dotenv import load_dotenv
load_dotenv() 

# DEBUG: Fjern disse linjer igen efter test!
print("DEBUG: TELEGRAM_TOKEN =", os.getenv("TELEGRAM_TOKEN"))
print("DEBUG: TELEGRAM_CHAT_ID =", os.getenv("TELEGRAM_CHAT_ID"))

def main_trading_cycle():
    print("✅ Botten starter trading-cyklus...")
    print("# Her indsætter du din logik for:")
    # print("# - Datadownload")
    # print("# - Feature engineering")
    # print("# - Modeltræning / prediction")
    # print("# - Signalberegning / trading")
    # print("# - Logging & Telegram")
    print("Her kommer trading-logikken!")
    time.sleep(2)

if __name__ == "__main__":
    print("✅ AI Trading Bot starter...")
    error_msg = None
    backup_path = None
    telegram_sent = False

    try:
        main_trading_cycle()

        # Backup efter cyklus (kræver at backup.py understøtter keep_days og keep_per_day)
        backup_path = make_backup(
            keep_days=7,         # Gem fx 7 dage (kan justeres)
            keep_per_day=10      # Max 10 backups per dag
        )
        print(f"✅ Backup gemt: {backup_path}")

        # Telegram-besked om succesfuld backup
        send_telegram_message(f"✅ Bot kørte OK og lavede backup: {backup_path}")
        telegram_sent = True

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Fejl under kørsel: {e}")

        # Telegram-besked om fejl
        try:
            send_telegram_message(f"❌ Bot FEJLEDE under kørsel: {e}")
            telegram_sent = True
        except Exception as tel_e:
            print(f"❌ Telegram FEJL: {tel_e}")

    finally:
        # Opdater BotStatus.md uanset succes/fejl
        update_bot_status(
            status="✅ Succes" if error_msg is None else "❌ Fejl",
            backup_path=backup_path,
            error_msg=error_msg if error_msg else "Ingen"
        )

        # Opdater CHANGELOG.md
        if error_msg is None:
            append_to_changelog(f"✅ Bot kørte og lavede backup: {backup_path}")
        else:
            append_to_changelog(f"❌ Bot fejlede: {error_msg}")

        # Ekstra: Telegram failsafe (hvis ikke allerede sendt pga. fejl)
        if not telegram_sent:
            try:
                msg = f"⚠️ Bot færdig, men Telegram-besked blev ikke sendt automatisk. (backup_path: {backup_path})"
                send_telegram_message(msg)
            except Exception:
                pass

    print("✅ Bot-kørsel færdig.")
