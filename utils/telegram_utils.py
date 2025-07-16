import os
import requests
import datetime
from dotenv import load_dotenv

try:
    from utils.plot_utils import generate_trend_graph
except ImportError:
    generate_trend_graph = None

load_dotenv()
LOG_PATH = "telegram_log.txt"

def telegram_enabled():
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or token.lower() in ("", "none", "dummy_token"):
        return False
    if not chat_id or chat_id.lower() in ("", "none", "dummy_id"):
        return False
    return True

def log_telegram(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{t}] {msg}\n")
    except Exception:
        print(f"[ADVARSEL] Telegram-log fejlede: {msg}")

def send_message(msg, chat_id=None, parse_mode=None, silent=False):
    """Send tekstbesked til Telegram."""
    log_telegram(f"Sender besked: {msg}")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_enabled():
        if not silent:
            print(f"🔕 [CI/test] Ville have sendt Telegram-besked: {msg}")
        log_telegram("[TESTMODE] Besked ikke sendt – Telegram inaktiv")
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
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

send_telegram_message = send_message  # Alias

def send_image(photo_path, caption="", chat_id=None, silent=False):
    """Send billede (photo) til Telegram."""
    log_telegram(f"Sender billede: {photo_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_enabled():
        if not silent:
            print(f"🔕 [CI/test] Ville have sendt billede: {photo_path} (caption: {caption})")
        log_telegram("[TESTMODE] Billede ikke sendt – Telegram inaktiv")
        return None
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
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

def send_document(doc_path, caption="", chat_id=None, silent=False):
    """Send dokument (fx CSV eller PDF) til Telegram."""
    log_telegram(f"Sender dokument: {doc_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_enabled():
        if not silent:
            print(f"🔕 [CI/test] Ville have sendt dokument: {doc_path} (caption: {caption})")
        log_telegram("[TESTMODE] Dokument ikke sendt – Telegram inaktiv")
        return None
    url = f"https://api.telegram.org/bot{token}/sendDocument"
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
    """Send hjertelyd/’ping’ til Telegram (viser at botten kører)."""
    t = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"💓 Botten kører stadig! ({t})"
    send_message(msg, chat_id=chat_id)
    log_telegram("Heartbeat sendt.")

def send_performance_report(metrics, symbol="", timeframe="", window=None, chat_id=None):
    """Send performance-summary til Telegram – HTML-format tilpasset."""
    msg = f"<b>📊 Performance Report {symbol} {timeframe} {window or ''}</b>\n"
    msg += f"Sharpe: <b>{metrics.get('sharpe', 0):.2f}</b> | Calmar: <b>{metrics.get('calmar', 0):.2f}</b> | Sortino: <b>{metrics.get('sortino', 0):.2f}</b>\n"
    msg += f"Volatilitet: <b>{metrics.get('volatility', 0):.2f}</b>\n"
    msg += f"Win-rate: <b>{metrics.get('win_rate', 0):.1f}%</b> | Profit faktor: <b>{metrics.get('profit_factor', 0):.2f}</b>\n"
    msg += f"Kelly: <b>{metrics.get('kelly_criterion', 0):.2f}</b> | Expectancy: <b>{metrics.get('expectancy', 0):.2f}%</b>\n"
    msg += f"Profit: <b>{metrics.get('abs_profit', 0):.2f} ({metrics.get('pct_profit', 0):.2f}%)</b>\n"
    msg += f"Max Drawdown: <b>{metrics.get('max_drawdown', 0):.2%}</b>\n"
    msg += f"Antal handler: <b>{metrics.get('total_trades', 0)}</b>\n"
    msg += f"Bedste: <b>{metrics.get('best_trade', 0):.2f}%</b> | Værste: <b>{metrics.get('worst_trade', 0):.2f}%</b>\n"
    send_message(msg, chat_id=chat_id, parse_mode="HTML")
    log_telegram("Performance report sendt til Telegram.")

def send_strategy_metrics(metrics, chat_id=None):
    """Send strategi-metrics i tekstformat til Telegram."""
    msg = (
        f"Strategi-metrics:\n"
        f"Profit: {metrics.get('profit_pct', 0):.2f}%\n"
        f"Win-rate: {metrics.get('win_rate', 0):.1f}%\n"
        f"Drawdown: {metrics.get('drawdown_pct', 0):.2f}%\n"
        f"Sharpe: {metrics.get('sharpe', 'N/A')}\n"
        f"Antal handler: {metrics.get('num_trades', 0)}"
    )
    send_message(msg, chat_id=chat_id)
    log_telegram("Strategi-metrics sendt.")

def send_ensemble_metrics(metrics, details=None, chat_id=None):
    """Send ensemble (voting)-metrics inkl. detaljer til Telegram."""
    msg = "🤖 <b>Ensemble/voting performance</b>\n"
    msg += f"Test accuracy: <b>{metrics.get('ensemble_test_acc', 0):.2%}</b>\n"
    msg += f"ML test: <b>{metrics.get('ml_test_acc', 0):.2%}</b> | DL test: <b>{metrics.get('dl_test_acc', 0):.2%}</b>\n"
    msg += f"Train ML: <b>{metrics.get('ml_train_acc', 0):.2%}</b> | Val ML: <b>{metrics.get('ml_val_acc', 0):.2%}</b>\n"
    if details:
        msg += f"\n<b>Details:</b>\n{details}\n"
    send_message(msg, parse_mode="HTML", chat_id=chat_id)
    log_telegram("Ensemble metrics sendt.")

def send_auto_status_summary(summary_text, image_path=None, doc_path=None, chat_id=None):
    """Send samlet status, evt. med graf/dokument."""
    send_message(summary_text, chat_id=chat_id)
    if image_path and os.path.exists(image_path):
        send_image(image_path, caption="📈 Equity Curve", chat_id=chat_id)
    if doc_path and os.path.exists(doc_path):
        send_document(doc_path, caption="📊 Trade Journal", chat_id=chat_id)

def send_trend_graph(chat_id=None, history_path="outputs/performance_history.csv", img_path="outputs/balance_trend.png", caption="📈 Balanceudvikling"):
    """Generér og send balance-trend-graf til Telegram."""
    try:
        if generate_trend_graph:
            img_path = generate_trend_graph(history_path=history_path, img_path=img_path)
            if img_path and os.path.exists(img_path):
                send_image(img_path, caption=caption, chat_id=chat_id)
            else:
                send_message("Kunne ikke generere balance-trend-graf.", chat_id=chat_id)
        else:
            send_message("Plot-utils ikke tilgængelig – trend-graf ikke genereret.", chat_id=chat_id)
    except Exception as e:
        print(f"❌ Fejl ved trend-graf: {e}")
        log_telegram(f"EXCEPTION ved send_trend_graph: {e}")
        send_message(f"❌ Fejl ved generering/sending af trend-graf: {e}", chat_id=chat_id)

# Testfunktion
if __name__ == "__main__":
    send_message("Testbesked fra din AI trading bot!")
    send_telegram_heartbeat()
    test_metrics = {
        "sharpe": 1.12, "calmar": 0.95, "sortino": 1.34, "volatility": 2.5, "win_rate": 43.2,
        "profit_factor": 1.45, "kelly_criterion": 0.18, "expectancy": 1.3, "abs_profit": 1234,
        "pct_profit": 12.3, "max_drawdown": -0.22, "total_trades": 84, "best_trade": 5.1, "worst_trade": -4.4
    }
    send_performance_report(test_metrics, symbol="BTCUSDT", timeframe="1h", window="0-200")
    send_trend_graph()
