import os
import requests
import datetime

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
LOG_PATH = "telegram_log.txt"  # Simpel logfil i projektroden

def telegram_enabled():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.lower() in ("", "none", "dummy_token"):
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID.lower() in ("", "none", "dummy_id"):
        return False
    return True

def log_telegram(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{t}] {msg}\n")

def send_telegram_message(msg, parse_mode=None):
    log_telegram(f"Sender besked: {msg}")
    if not telegram_enabled():
        print(f"🔕 [CI/test] Ville have sendt Telegram-besked: {msg}")
        log_telegram("[TESTMODE] Besked ikke sendt – Telegram inaktiv")
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
            log_telegram("Besked sendt OK.")
        else:
            print(f"❌ Telegram-fejl: {resp.text}")
            log_telegram(f"FEJL ved sendMessage: {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception: {e}")
        log_telegram(f"EXCEPTION ved sendMessage: {e}")

def send_telegram_photo(photo_path, caption=""):
    log_telegram(f"Sender billede: {photo_path} (caption: {caption})")
    if not telegram_enabled():
        print(f"🔕 [CI/test] Ville have sendt billede: {photo_path} (caption: {caption})")
        log_telegram("[TESTMODE] Billede ikke sendt – Telegram inaktiv")
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
            log_telegram("Billede sendt OK.")
        else:
            print(f"❌ Telegram-fejl (billede): {resp.text}")
            log_telegram(f"FEJL ved sendPhoto: {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception (billede): {e}")
        log_telegram(f"EXCEPTION ved sendPhoto: {e}")

def send_telegram_document(doc_path, caption=""):
    log_telegram(f"Sender dokument: {doc_path} (caption: {caption})")
    if not telegram_enabled():
        print(f"🔕 [CI/test] Ville have sendt dokument: {doc_path} (caption: {caption})")
        log_telegram("[TESTMODE] Dokument ikke sendt – Telegram inaktiv")
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
            log_telegram("Dokument sendt OK.")
        else:
            print(f"❌ Telegram-fejl (dokument): {resp.text}")
            log_telegram(f"FEJL ved sendDocument: {resp.text}")
    except Exception as e:
        print(f"❌ Telegram exception (dokument): {e}")
        log_telegram(f"EXCEPTION ved sendDocument: {e}")

def send_telegram_heartbeat():
    t = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"💓 Botten kører stadig! ({t})"
    send_telegram_message(msg)
    log_telegram("Heartbeat sendt.")

def send_strategy_metrics(metrics):
    msg = (
        f"Strategi-metrics:\n"
        f"Profit: {metrics.get('profit_pct', 0)}%\n"
        f"Win-rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
        f"Drawdown: {metrics.get('drawdown_pct', 0)}%\n"
        f"Antal handler: {metrics.get('num_trades', 0)}"
    )
    send_telegram_message(msg)
    log_telegram("Strategi-metrics sendt.")

def send_status_advarsel(metrics, threshold=0.3):
    if metrics.get("win_rate", 1) < threshold:
        send_telegram_message(
            f"⚠️ Advarsel: Win-rate er lav ({metrics['win_rate']*100:.1f}%) – check strategi!"
        )
        log_telegram("Advarsel om lav win-rate sendt.")

# --- Testfunktion ---
if __name__ == "__main__":
    send_telegram_message("Testbesked fra din AI trading bot!")
    send_telegram_heartbeat()
    # send_telegram_photo("graphs/btc_balance_20250605.png", caption="Balanceudvikling")
    # send_telegram_document("data/trades.csv", caption="Trade journal")
