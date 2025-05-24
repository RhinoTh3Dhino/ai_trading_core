import time
from utils.backup import make_backup
from utils.botstatus import update_bot_status
from utils.changelog import append_to_changelog

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
    try:
        main_trading_cycle()

        # Brug de nye parametre – dato-struktur og fleksibel oprydning
        backup_path = make_backup(
            keep_days=7,         # Gem fx 7 dage (kan justeres)
            keep_per_day=10      # Max 10 backups per dag
        )
        print(f"✅ Backup gemt: {backup_path}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Fejl under kørsel: {e}")
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

    print("✅ Bot-kørsel færdig.")
