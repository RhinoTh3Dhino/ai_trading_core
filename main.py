import time
from utils.backup import make_backup
from utils.botstatus import update_bot_status   # Tilføj denne import

def main_trading_cycle():
    print("✅ Botten starter trading-cyklus...")
    # Her indsætter du din logik for:
    # - Datadownload
    # - Feature engineering
    # - Modeltræning / prediction
    # - Signalberegning / trading
    # - Logging & Telegram
    print("Her kommer trading-logikken!")
    time.sleep(2)

if __name__ == "__main__":
    print("✅ AI Trading Bot starter...")
    error_msg = None
    backup_path = None
    try:
        main_trading_cycle()
        # Backup efter cyklus (kan også placeres før/efter bestemte trin)
        backup_path = make_backup(keep_last=10)
        print(f"✅ Backup gemt: {backup_path}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Fejl under kørsel: {e}")
    finally:
        # Opdater BotStatus.md uanset succes/fejl
        update_bot_status(
            status="✅ Succes" if error_msg is None else "❌ Fejl",
            backup_path=backup_path,
            error_msg=error_msg
        )
        print("✅ Bot-kørsel færdig.")
