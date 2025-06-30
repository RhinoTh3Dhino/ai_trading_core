import os
import requests
import datetime

from dotenv import load_dotenv
load_dotenv()  # Hent variabler fra .env hvis ikke allerede sat

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
LOG_PATH = "telegram_log.txt"

def telegram_enabled():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.lower() in ("", "none", "dummy_token"):
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID.lower() in ("", "none", "dummy_id"):
        return False
    return True

def log_telegram(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{t}] {msg}\n")
    except Exception:
        pass  # Logger må aldrig stoppe botten

def send_message(msg, chat_id=None, parse_mode=None):
    log_telegram(f"Sender besked: {msg}")
    if not telegram_enabled():
        print(f"🔕 [CI/test] Ville have sendt Telegram-besked: {msg}")
        log_telegram("[TESTMODE] Besked ikke sendt – Telegram inaktiv")
        return None
    _chat_id = int(chat_id) if chat_id is not None else int(TELEGRAM_CHAT_ID)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": _chat_id, "text": msg}
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
        return resp
    except Exception as e:
        print(f"❌ Telegram exception: {e}")
        log_telegram(f"EXCEPTION ved sendMessage: {e}")
        return None

send_telegram_message = send_message

def send_image(photo_path, caption="", chat_id=None):
    log_telegram(f"Sender billede: {photo_path} (caption: {caption})")
    if not telegram_enabled():
        print(f"🔕 [CI/test] Ville have sendt billede: {photo_path} (caption: {caption})")
        log_telegram("[TESTMODE] Billede ikke sendt – Telegram inaktiv")
        return None
    _chat_id = int(chat_id) if chat_id is not None else int(TELEGRAM_CHAT_ID)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    data = {"chat_id": _chat_id, "caption": caption}
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
        return resp
    except Exception as e:
        print(f"❌ Telegram exception (billede): {e}")
        log_telegram(f"EXCEPTION ved sendPhoto: {e}")
        return None

def send_document(doc_path, caption="", chat_id=None):
    log_telegram(f"Sender dokument: {doc_path} (caption: {caption})")
    if not telegram_enabled():
        print(f"🔕 [CI/test] Ville have sendt dokument: {doc_path} (caption: {caption})")
        log_telegram("[TESTMODE] Dokument ikke sendt – Telegram inaktiv")
        return None
    _chat_id = int(chat_id) if chat_id is not None else int(TELEGRAM_CHAT_ID)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    data = {"chat_id": _chat_id, "caption": caption}
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
        return resp
    except Exception as e:
        print(f"❌ Telegram exception (dokument): {e}")
        log_telegram(f"EXCEPTION ved sendDocument: {e}")
        return None

def send_telegram_heartbeat(chat_id=None):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"💓 Botten kører stadig! ({t})"
    send_message(msg, chat_id=chat_id)
    log_telegram("Heartbeat sendt.")

def send_strategy_metrics(metrics, chat_id=None):
    msg = (
        f"Strategi-metrics:\n"
        f"Profit: {metrics.get('profit_pct', 0):.2f}%\n"
        f"Win-rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
        f"Drawdown: {metrics.get('drawdown_pct', 0):.2f}%\n"
        f"Sharpe: {metrics.get('sharpe', 'N/A')}\n"
        f"Antal handler: {metrics.get('num_trades', 0)}"
    )
    send_message(msg, chat_id=chat_id)
    log_telegram("Strategi-metrics sendt.")

def send_status_advarsel(metrics, threshold=0.3, chat_id=None):
    if metrics.get("win_rate", 1) < threshold:
        send_message(
            f"⚠️ Advarsel: Win-rate er lav ({metrics['win_rate']*100:.1f}%) – check strategi!",
            chat_id=chat_id
        )
        log_telegram("Advarsel om lav win-rate sendt.")

def send_regime_warning(regime_stats, threshold=0.3, chat_id=None):
    if not regime_stats: return
    for regime, stats in regime_stats.items():
        if stats.get("win_rate", 1) < threshold:
            msg = (
                f"⚠️ ADVARSEL: Win-rate lav i regime '{regime}': {stats['win_rate']*100:.1f}%\n"
                f"Antal handler: {stats['num_trades']} | Profit: {stats['profit_pct']}%"
            )
            send_message(msg, chat_id=chat_id)
            log_telegram(f"Regime-advarsel sendt for {regime}.")

def send_regime_summary(regime_stats, chat_id=None):
    if not regime_stats:
        send_message("Ingen regime-stats tilgængelig.", chat_id=chat_id)
        return
    lines = ["📊 Performance pr. regime:"]
    for regime, stats in regime_stats.items():
        lines.append(
            f"{regime}: Win-rate {stats['win_rate']*100:.1f}%, "
            f"Profit {stats['profit_pct']}%, "
            f"Trades {stats['num_trades']}"
        )
    send_message("\n".join(lines), chat_id=chat_id)
    log_telegram("Regime-summary sendt.")

# Testfunktion
if __name__ == "__main__":
    send_message("Testbesked fra din AI trading bot!")
    send_telegram_heartbeat()

    test_stats = {
        "bull": {"win_rate": 0.35, "profit_pct": 2.4, "num_trades": 10},
        "bear": {"win_rate": 0.25, "profit_pct": -1.2, "num_trades": 5},
        "neutral": {"win_rate": 0.4, "profit_pct": 0.0, "num_trades": 8}
    }
    send_regime_summary(test_stats)
    send_regime_warning(test_stats, threshold=0.3)
