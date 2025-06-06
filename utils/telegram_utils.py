import os
import requests
import datetime

# Hent credentials fra miljøvariabler
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(msg, parse_mode=None):
    """Sender en tekstbesked til din Telegram-bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram ikke konfigureret (TELEGRAM_BOT_TOKEN/CHAT_ID mangler).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": int(TELEGRAM_CHAT_ID),
        "text": msg
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.ok:
            print("✅ Telegram-besked sendt!")
        else:
            print(f"❌ Telegram-fejl: {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception: {e}")

def send_telegram_photo(photo_path, caption=""):
    """Sender et billede (fx graf) til Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram ikke konfigureret.")
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

def send_telegram_document(doc_path, caption=""):
    """Sender et dokument/CSV til Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram ikke konfigureret.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    data = {
        "chat_id": int(TELEGRAM_CHAT_ID),
        "caption": caption
    }
    try:
        with open(doc_path, "rb") as doc:
            files = {"document": doc}
            resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.ok:
            print("✅ Telegram-dokument sendt!")
        else:
            print(f"❌ Telegram-fejl (dokument): {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception (dokument): {e}")

def send_telegram_heartbeat():
    """Sender en 'hjertelyd' for at vise at botten kører."""
    t = datetime.datetime.now().strftime("%H:%M:%S")
    send_telegram_message(f"💓 Botten kører stadig! ({t})")

def send_strategy_metrics(metrics):
    """Sender strategi-metrics som samlet besked."""
    msg = (
        f"Strategi-metrics:\n"
        f"Profit: {metrics.get('profit_pct', 0)}%\n"
        f"Win-rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
        f"Drawdown: {metrics.get('drawdown_pct', 0)}%\n"
        f"Antal handler: {metrics.get('num_trades', 0)}"
    )
    send_telegram_message(msg)

def send_status_advarsel(metrics, threshold=0.3):
    """Sender advarsel hvis win-rate er under threshold."""
    if metrics.get("win_rate", 1) < threshold:
        send_telegram_message(
            f"⚠️ Advarsel: Win-rate er lav ({metrics['win_rate']*100:.1f}%) – check strategi!"
        )

# --- Testfunktion ---
if __name__ == "__main__":
    send_telegram_message("Testbesked fra din AI trading bot!")
    send_telegram_heartbeat()
    # send_telegram_photo("graphs/btc_balance_20250605.png", caption="Balanceudvikling")
    # send_telegram_document("data/trades.csv", caption="Trade journal")
