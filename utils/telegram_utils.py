import os
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(msg):
    """Sender en besked til din Telegram-bot"""
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
