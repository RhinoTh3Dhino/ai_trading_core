import os
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(msg):
    """Sender en tekstbesked til din Telegram-bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram ikke konfigureret.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": int(TELEGRAM_CHAT_ID),
        "text": msg,
    }
    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.ok:
            print("✅ Telegram-besked sendt!")
        else:
            print(f"❌ Telegram-fejl: {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception: {e}")

def send_telegram_photo(photo_path, caption=""):
    """Sender et billede (fx graf) til din Telegram-bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram ikke konfigureret.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    data = {
        "chat_id": int(TELEGRAM_CHAT_ID),
        "caption": caption
    }
    try:
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.ok:
            print("✅ Telegram-billede sendt!")
        else:
            print(f"❌ Telegram-fejl (billede): {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception (billede): {e}")
