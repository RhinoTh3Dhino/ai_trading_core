# tests/test_log_utils.py

from utils.log_utils import log_device_status
from utils.telegram_utils import send_message  # hvis du vil teste Telegram

def test_logging():
    # Test: Logging til BotStatus.md, logfil og (valgfrit) Telegram
    log_device_status(
        context="test_log_utils",
        extra={"strategy": "ensemble", "feature_file": "dummy.csv", "param": 123},
        telegram_func=send_message,   # Fjern hvis du ikke vil teste Telegram
        print_console=True
    )

if __name__ == "__main__":
    test_logging()
    print("✅ log_utils.py test kørt.")
