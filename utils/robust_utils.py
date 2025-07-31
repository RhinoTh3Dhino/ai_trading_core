import logging
import os
import traceback
from utils.telegram_utils import send_telegram_message

# Opret log-mappen hvis den ikke findes
os.makedirs("logs", exist_ok=True)

# Sæt logging op: gem ALT under ERROR i logs/errors.log
logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def safe_run(func, *args, **kwargs):
    """
    Kør enhver funktion med robust fejlhåndtering, auto-logging og Telegram-besked.
    Sender traceback til Telegram og logger fejlen i logs/errors.log.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        tb_str = traceback.format_exc()
        msg = f"❌ Fejl i {func.__name__}: {e}\n" f"Traceback:\n{tb_str}"
        # Telegram beskeder må max være 4096 tegn
        send_telegram_message(msg[:4000])
        logging.error(msg)
        print(msg)
        return None
