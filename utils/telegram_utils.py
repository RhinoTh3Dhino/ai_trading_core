from utils.project_path import PROJECT_ROOT  # AUTO PATH CONVERTED

import os
import requests
import datetime
from dotenv import load_dotenv

# === Importér monitoring_utils for live-metrics og alarmer ===
from utils.monitoring_utils import (
    calculate_live_metrics,
    check_drawdown_alert,
    check_winrate_alert,
    check_profit_alert,
)

try:
    from utils.plot_utils import generate_trend_graph
except ImportError:
    generate_trend_graph = None

load_dotenv()
LOG_PATH = PROJECT_ROOT / "telegram_log.txt"  # AUTO PATH CONVERTED

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
    log_telegram(f"Sender besked: {msg}")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_enabled():
        if not silent:
            print(f"[TESTMODE] Ville have sendt Telegram-besked: {msg}")
        log_telegram("[TESTMODE] Besked ikke sendt – Telegram inaktiv")
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": _chat_id, "text": msg}
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.ok:
            print("[OK] Telegram-besked sendt!")
            log_telegram("Besked sendt OK.")
        else:
            print(f"[FEJL] Telegram-fejl: {resp.text}")
            log_telegram(f"FEJL ved sendMessage: {resp.text}")
        return resp
    except Exception as e:
        print(f"[FEJL] Telegram exception: {e}")
        log_telegram(f"EXCEPTION ved sendMessage: {e}")
        return None

send_telegram_message = send_message  # Alias

def send_image(photo_path, caption="", chat_id=None, silent=False):
    log_telegram(f"Sender billede: {photo_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_enabled():
        if not silent:
            print(f"[TESTMODE] Ville have sendt billede: {photo_path} (caption: {caption})")
        log_telegram("[TESTMODE] Billede ikke sendt – Telegram inaktiv")
        return None
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    data = {"chat_id": _chat_id, "caption": caption}
    try:
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.ok:
            print("[OK] Telegram-billede sendt!")
            log_telegram("Billede sendt OK.")
        else:
            print(f"[FEJL] Telegram-fejl (billede): {resp.text}")
            log_telegram(f"FEJL ved sendPhoto: {resp.text}")
        return resp
    except Exception as e:
        print(f"[FEJL] Telegram exception (billede): {e}")
        log_telegram(f"EXCEPTION ved sendPhoto: {e}")
        return None

def send_document(doc_path, caption="", chat_id=None, silent=False):
    log_telegram(f"Sender dokument: {doc_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_enabled():
        if not silent:
            print(f"[TESTMODE] Ville have sendt dokument: {doc_path} (caption: {caption})")
        log_telegram("[TESTMODE] Dokument ikke sendt – Telegram inaktiv")
        return None
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    data = {"chat_id": _chat_id, "caption": caption}
    try:
        with open(doc_path, "rb") as doc:
            files = {"document": doc}
            resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.ok:
            print("[OK] Telegram-dokument sendt!")
            log_telegram("Dokument sendt OK.")
        else:
            print(f"[FEJL] Telegram-fejl (dokument): {resp.text}")
            log_telegram(f"FEJL ved sendDocument: {resp.text}")
        return resp
    except Exception as e:
        print(f"[FEJL] Telegram exception (dokument): {e}")
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
        f"Win-rate: {metrics.get('win_rate', 0):.1f}%\n"
        f"Drawdown: {metrics.get('drawdown_pct', 0):.2f}%\n"
        f"Sharpe: {metrics.get('sharpe', 'N/A')}\n"
        f"Antal handler: {metrics.get('num_trades', 0)}"
    )
    send_message(msg, chat_id=chat_id)
    log_telegram("Strategi-metrics sendt.")

def send_auto_status_summary(summary_text, image_path=None, doc_path=None, chat_id=None):
    send_message(summary_text, chat_id=chat_id)
    if image_path and os.path.exists(image_path):
        send_image(image_path, caption="📈 Equity Curve", chat_id=chat_id)
    if doc_path and os.path.exists(doc_path):
        send_document(doc_path, caption="📊 Trade Journal", chat_id=chat_id)

def send_trend_graph(
    chat_id=None,
    history_path=PROJECT_ROOT / "outputs" / "performance_history.csv",  # AUTO PATH CONVERTED
    img_path=PROJECT_ROOT / "outputs" / "balance_trend.png",            # AUTO PATH CONVERTED
    caption="📈 Balanceudvikling"
):
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
        print(f"[FEJL] Fejl ved trend-graf: {e}")
        log_telegram(f"EXCEPTION ved send_trend_graph: {e}")
        send_message(f"Fejl ved generering/sending af trend-graf: {e}", chat_id=chat_id)

def send_live_metrics(trades_df, balance_df, symbol="", timeframe="", thresholds=None, chat_id=None):
    """
    Send live performance-metrics og alarmer til Telegram.
    thresholds: dict, fx {"drawdown": -20, "winrate": 20, "profit": -10}
    """
    metrics = calculate_live_metrics(trades_df, balance_df)
    msg = (
        f"📡 <b>Live trading-status {symbol} {timeframe}</b>\n"
        f"Profit: <b>{metrics['profit_pct']:.2f}%</b>\n"
        f"Win-rate: <b>{metrics['win_rate']:.1f}%</b>\n"
        f"Drawdown: <b>{metrics['drawdown_pct']:.2f}%</b>\n"
        f"Antal handler: <b>{metrics['num_trades']}</b>\n"
        f"Profit factor: <b>{metrics['profit_factor']}</b>\n"
        f"Sharpe: <b>{metrics['sharpe']}</b>\n"
    )
    send_message(msg, chat_id=chat_id, parse_mode="HTML")
    alarm_msgs = []
    if thresholds:
        if check_drawdown_alert(metrics, threshold=thresholds.get("drawdown", -20)):
            alarm_msgs.append(f"🚨 ADVARSEL: Max drawdown under {thresholds.get('drawdown', -20)}%! ({metrics['drawdown_pct']:.2f}%)")
        if check_winrate_alert(metrics, threshold=thresholds.get("winrate", 20)):
            alarm_msgs.append(f"🚨 ADVARSEL: Win-rate under {thresholds.get('winrate', 20)}%! ({metrics['win_rate']:.1f}%)")
        if check_profit_alert(metrics, threshold=thresholds.get("profit", -10)):
            alarm_msgs.append(f"🚨 ADVARSEL: Profit under {thresholds.get('profit', -10)}%! ({metrics['profit_pct']:.2f}%)")
    if alarm_msgs:
        for alarm in alarm_msgs:
            send_message(alarm, chat_id=chat_id)
            log_telegram(alarm)

# Testfunktion
if __name__ == "__main__":
    send_message("Testbesked fra din AI trading bot!")
    send_telegram_heartbeat()
    import pandas as pd
    # Dummy for live-metrics test
    balance_df = pd.DataFrame({"balance": [1000, 980, 950, 990, 970, 1005]})
    trades_df = pd.DataFrame({
        "type": ["BUY", "TP", "BUY", "SL", "BUY", "TP", "SELL", "TP", "SELL", "SL"],
        "profit": [0, 0.02, 0, -0.015, 0, 0.01, 0, 0.03, 0, -0.012]
    })
    send_live_metrics(trades_df, balance_df, symbol="BTCUSDT", timeframe="1h", thresholds={"drawdown": -2, "winrate": 60, "profit": -1})
    send_trend_graph()
