# main.py
import time
from backup import make_backup

def main_trading_cycle():
    print("🚀 Botten starter trading-cyklus...")
    # Her indsætter du din logik for:
    # - Datadownload
    # - Feature engineering
    # - Modeltræning / prediction
    # - Signalberegning / trading
    # - Logging & Telegram
    print("Her kommer trading-logikken!")
    # Simuler fx. med sleep(2)
    time.sleep(2)

if __name__ == "__main__":
    print("✅ AI Trading Bot starter...")
    try:
        main_trading_cycle()
        # Backup efter cyklus (kan også placeres før/efter bestemte trin)
        backup_path = make_backup(keep_last=10)
        print(f"🗄️ Backup gemt: {backup_path}")
        # (Ekstra: Send evt. Telegrambesked om backup her)
    except Exception as e:
        print(f"❌ Fejl under kørsel: {e}")
    print("✅ Bot-kørsel færdig.")
