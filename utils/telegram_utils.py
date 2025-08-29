from utils.project_path import PROJECT_ROOT

import os
import time
import requests
import datetime
import html
from dotenv import load_dotenv
from typing import Optional

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
# AUTO PATH CONVERTED
LOG_PATH = PROJECT_ROOT / "telegram_log.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Terminal-verbosity (sæt TELEGRAM_VERBOSE=0 for at dæmpe "[OK] Telegram-... sendt!")
VERBOSE = os.getenv("TELEGRAM_VERBOSE", "1").lower() not in ("0", "false", "no", "off")

# Telegram begrænsninger (konservative)
_MAX_TEXT = 4096        # sendMessage max tekstlængde
_MAX_CAPTION = 1024     # caption max (foto/dokument)

# ------------------------------------------------------------
# "Ro på" – ENV-styrede parametre
# ------------------------------------------------------------
def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

DEDUPE_TTL_SEC = _get_int_env("TELEGRAM_DEDUPE_TTL_SEC", 120)           # samme besked-nøgle under ttl => undertrykkes
COOLDOWN_GLOBAL_SEC = _get_int_env("TELEGRAM_COOLDOWN_GLOBAL_SEC", 5)   # minimum afstand mellem beskeder globalt
COOLDOWN_PER_SYMBOL_SEC = _get_int_env("TELEGRAM_COOLDOWN_PER_SYMBOL_SEC", 15)
BATCH_LOWPRIO_EVERY_SEC = _get_int_env("TELEGRAM_BATCH_LOWPRIO_EVERY_SEC", 60)
BATCH_MAX_ITEMS = _get_int_env("TELEGRAM_BATCH_MAX_ITEMS", 20)
LOG_DECISIONS = _get_bool_env("TELEGRAM_LOG_DECISIONS", True)           # log “SUPPRESS/NOTIFY”-årsag

# Tilstand for gating/batching (in-memory)
_last_sent_global_ts: float = 0.0
_last_sent_by_symbol: dict[str, float] = {}
_dedupe_store: dict[str, float] = {}
_lowprio_buffer: list[tuple[float, str, Optional[str]]] = []  # (ts, text, symbol)
_last_batch_flush_ts: float = 0.0


# ------------------------------------------------------------
# Små helpers til print/log
# ------------------------------------------------------------
def _vprint(msg: str, silent: bool = False):
    if VERBOSE and not silent:
        print(msg)


def _eprint(msg: str):
    # fejl bør altid kunne ses
    print(msg)


def _now_ts() -> float:
    return time.time()


def _decision_log(action: str, reason: str, *, symbol: Optional[str] = None, extra: str = ""):
    if not LOG_DECISIONS:
        return
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sym = symbol or "-"
    log_telegram(f"[DECISION] {stamp} | {action} | {reason} | symbol={sym} {extra}")


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
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
        _eprint(f"[ADVARSEL] Telegram-log fejlede: {msg}")


def _chunks(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i : i + n]


_MD2_SPECIALS = set(r'_*[]()~`>#+-=|{}.!')


def _escape_markdown_v2(s: str) -> str:
    # Escape alle Telegram MarkdownV2 specials ved at prefixe \
    out = []
    for ch in s:
        if ch in _MD2_SPECIALS:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _as_html_pre(s: str) -> str:
    # Sikker HTML: vis ALT som monospaced tekst i <pre>…</pre>
    return f"<pre>{html.escape(s)}</pre>"


def _is_parse_entities_error(resp_json) -> bool:
    try:
        if isinstance(resp_json, dict) and resp_json.get("ok") is False:
            desc = (resp_json.get("description") or "").lower()
            return "can't parse entities" in desc
    except Exception:
        pass
    return False


def _resp_ok(resp):
    # Robust ok-detektion (også til mocks i tests)
    try:
        js = resp.json()
    except Exception:
        js = None

    if isinstance(js, dict) and "ok" in js:
        return bool(js.get("ok", False)), js
    status = getattr(resp, "status_code", None)
    ok_flag = status is not None and 200 <= int(status) < 300
    return ok_flag, js


# ------------------------------------------------------------
# Tekst-beskeder (rå send)
# ------------------------------------------------------------
def _send_text_request(token: str, chat_id: str, text: str, parse_mode: str | None):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return requests.post(url, json=payload, timeout=10)


def _send_text_chunked(token: str, chat_id: str, text: str, parse_mode: str | None):
    """
    Sender tekst i chunks og forsøger fallback ved entity-parse-fejl:
    - hvis parse_mode=Markdown/MarkdownV2: escaper vi først
    - hvis parse_mode=HTML: sender råt; ved parse-fejl -> fallback til HTML <pre>
    - hvis parse_mode=None: ren tekst
    """
    # For MarkdownV2: escape hele teksten først
    if parse_mode and parse_mode.upper().startswith("MARKDOWN"):
        text = _escape_markdown_v2(text)

    # Chunk-grænse (Telegram tæller tegn, ikke bytes)
    limit = _MAX_TEXT

    last_resp_json = None
    for chunk in _chunks(text, limit):
        resp = _send_text_request(token, chat_id, chunk, parse_mode)
        ok, resp_json = _resp_ok(resp)
        last_resp_json = resp_json if isinstance(resp_json, dict) else None
        if ok:
            continue  # næste chunk

        # Hvis det er en parsing-fejl, prøv sikker fallback (HTML <pre>)
        if _is_parse_entities_error(resp_json):
            safe = _as_html_pre(chunk)
            resp2 = _send_text_request(token, chat_id, safe, "HTML")
            ok2, resp2_json = _resp_ok(resp2)
            last_resp_json = resp2_json if isinstance(resp2_json, dict) else last_resp_json
            if not ok2:
                _eprint(f"[FEJL] Telegram parse & fallback fejlede: {getattr(resp2, 'text', '')}")
                log_telegram(f"Fallback parse-fejl: {getattr(resp2, 'text', '')}")
        else:
            # Anden fejl – log og fortsæt til næste chunk (for ikke at spamme)
            _eprint(f"[FEJL] Telegram API {getattr(resp, 'status_code', '?')}: {getattr(resp, 'text', '')}")
            log_telegram(f"FEJL ved sendMessage: {getattr(resp, 'text', '')}")

    return last_resp_json if isinstance(last_resp_json, dict) else True


def send_message(msg, chat_id=None, parse_mode=None, silent=False):
    """
    Offentlig helper:
      - chunker automatisk
      - håndterer MarkdownV2-escaping og fallback til HTML <pre> ved parse-fejl
      - bevarer din eksisterende signatur
    """
    log_telegram(f"Sender besked: {msg}")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")

    if not telegram_enabled():
        if not silent:
            _vprint(f"[TESTMODE] Ville have sendt Telegram-besked: {msg}", silent=False)
        log_telegram("[TESTMODE] Besked ikke sendt – Telegram inaktiv")
        return None

    try:
        resp_json_or_true = _send_text_chunked(token, _chat_id, msg, parse_mode)
        # Hvis vi nåede hertil uden exception, antag OK (evt. sidste svar-json)
        _vprint("[OK] Telegram-besked sendt!", silent=silent)
        log_telegram("Besked sendt OK.")
        return resp_json_or_true
    except Exception as e:
        _eprint(f"[FEJL] Telegram exception: {e}")
        log_telegram(f"EXCEPTION ved sendMessage: {e}")
        return None


send_telegram_message = send_message  # Alias


# ------------------------------------------------------------
# "Ro på" – gating API (dedupe, cooldown, batching)
# ------------------------------------------------------------
def _is_duplicate(key: str) -> bool:
    if not key:
        return False
    now = _now_ts()
    exp = _dedupe_store.get(key, 0.0)
    if now <= exp:
        return True
    _dedupe_store[key] = now + max(1, DEDUPE_TTL_SEC)
    return False


def _in_cooldown(symbol: Optional[str]) -> bool:
    now = _now_ts()
    if now - _last_sent_global_ts < COOLDOWN_GLOBAL_SEC:
        return True
    if symbol:
        last = _last_sent_by_symbol.get(symbol, 0.0)
        if now - last < COOLDOWN_PER_SYMBOL_SEC:
            return True
    return False


def _mark_sent(symbol: Optional[str]):
    global _last_sent_global_ts
    _last_sent_global_ts = _now_ts()
    if symbol:
        _last_sent_by_symbol[symbol] = _last_sent_global_ts


def send_signal_message(
    text: str,
    *,
    symbol: Optional[str] = None,
    dedupe_key: Optional[str] = None,
    priority: str = "high",               # "high" | "low"
    parse_mode: Optional[str] = None,
    chat_id: Optional[str] = None,
    silent: bool = False,
):
    """
    Gated Telegram-sender:
    - Duplikatfilter: samme dedupe_key under TTL → SUPPRESS
    - Cooldown globalt og pr. symbol → SUPPRESS
    - priority="low" → lægges i batch-buffer og flushes periodisk
    Returnerer:
      - dict med {"ok": True, "suppressed": True, "reason": "..."} ved suppression
      - Ellers svar fra send_message(...)
    """
    key = dedupe_key or f"{symbol or '-'}|{hash(text)}"

    # low-prio → buffer og evtl. flush
    if priority.lower() == "low":
        _lowprio_buffer.append((_now_ts(), text, symbol))
        _decision_log("SUPPRESS", "queued_lowprio", symbol=symbol, extra=f"(buffer={len(_lowprio_buffer)})")
        maybe_flush_lowprio_batch(chat_id=chat_id)
        return {"ok": True, "suppressed": True, "reason": "queued_lowprio"}

    # high-prio → apply dedupe + cooldown
    if _is_duplicate(key):
        _decision_log("SUPPRESS", "duplicate", symbol=symbol, extra=f"key={key}")
        return {"ok": True, "suppressed": True, "reason": "duplicate"}

    if _in_cooldown(symbol):
        _decision_log("SUPPRESS", "cooldown", symbol=symbol)
        return {"ok": True, "suppressed": True, "reason": "cooldown"}

    # send
    resp = send_message(text, chat_id=chat_id, parse_mode=parse_mode, silent=silent)
    _mark_sent(symbol)
    _decision_log("NOTIFY", "sent", symbol=symbol)
    return resp


def maybe_flush_lowprio_batch(chat_id: Optional[str] = None, header: str = "🔔 Lav-prio opsummering"):
    """
    Flusher lav-prio buffer hvis interval er gået. Returnerer True hvis der blev sendt noget.
    """
    global _last_batch_flush_ts
    now = _now_ts()
    if now - _last_batch_flush_ts < max(5, BATCH_LOWPRIO_EVERY_SEC):
        return False
    if not _lowprio_buffer:
        _last_batch_flush_ts = now
        return False

    # Saml items (respektér max for ikke at lave romaner)
    items = list(_lowprio_buffer)
    del _lowprio_buffer[:]
    _last_batch_flush_ts = now

    # Byg opsummering
    lines = [f"{header} • {len(items)} beskeder"]
    # limiter output, resten tælles
    used = 0
    for ts_, txt, sym in items[:BATCH_MAX_ITEMS]:
        stamp = datetime.datetime.fromtimestamp(ts_).strftime("%H:%M:%S")
        prefix = f"[{stamp}]"
        if sym:
            prefix += f" [{sym}]"
        lines.append(f"• {prefix} {txt}")
        used += 1
    if len(items) > used:
        lines.append(f"… og {len(items) - used} flere")

    text = "\n".join(lines)
    send_message(text, chat_id=chat_id, parse_mode=None, silent=True)
    _decision_log("NOTIFY", "batch_flush", extra=f"items={len(items)}")
    return True


# ------------------------------------------------------------
# Billeder
# ------------------------------------------------------------
def _post_multipart(url: str, data: dict, files: dict):
    return requests.post(url, data=data, files=files, timeout=20)


def _caption_and_mode(caption: str, parse_mode: str | None):
    """
    Normaliser caption og parse_mode:
    - None        => ren tekst
    - Markdown/MarkdownV2  => escape caption + brug 'MarkdownV2'
    - HTML        => send som givet (antag at afsender ved hvad der laves)
    """
    if not caption:
        return caption, None if parse_mode is None else parse_mode

    if parse_mode and parse_mode.upper().startswith("MARKDOWN"):
        return _escape_markdown_v2(caption), "MarkdownV2"
    if parse_mode and parse_mode.upper() == "HTML":
        return caption, "HTML"
    # default: plain text (ingen parse)
    return caption, None


def send_image(photo_path, caption="", chat_id=None, silent=False, parse_mode=None):
    log_telegram(f"Sender billede: {photo_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")

    if not telegram_enabled():
        if not silent:
            _vprint(f"[TESTMODE] Ville have sendt billede: {photo_path} (caption: {caption})", silent=False)
        log_telegram("[TESTMODE] Billede ikke sendt – Telegram inaktiv")
        return None

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Mangler fil: {photo_path}")

        cap, pmode = _caption_and_mode(caption, parse_mode)

        # Telegrams caption-limit
        if cap:
            cap = cap[:_MAX_CAPTION]

        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": _chat_id}
            if cap:
                data["caption"] = cap
            if pmode:
                data["parse_mode"] = pmode
            resp = _post_multipart(url, data=data, files=files)

        status = getattr(resp, "status_code", None)
        ok_flag = status is not None and 200 <= int(status) < 300

        if ok_flag:
            _vprint("[OK] Telegram-billede sendt!", silent=silent)
            log_telegram("Billede sendt OK.")
        else:
            detail = getattr(resp, "text", "")
            # Fallback: hvis parse-entities-fejl på caption, så prøv igen med HTML <pre>
            try:
                js = resp.json()
            except Exception:
                js = None
            if _is_parse_entities_error(js) and caption:
                safe_cap = _as_html_pre(caption)[:_MAX_CAPTION]
                with open(photo_path, "rb") as photo2:
                    files2 = {"photo": photo2}
                    data2 = {"chat_id": _chat_id, "caption": safe_cap, "parse_mode": "HTML"}
                    resp2 = _post_multipart(url, data=data2, files=files2)
                status2 = getattr(resp2, "status_code", None)
                ok2 = status2 is not None and 200 <= int(status2) < 300
                if ok2:
                    _vprint("[OK] Telegram-billede sendt (fallback)!", silent=silent)
                    log_telegram("Billede sendt OK (fallback).")
                    return resp2
            _eprint(f"[FEJL] Telegram-fejl (billede): {detail}")
            log_telegram(f"FEJL ved sendPhoto: {detail}")
        return resp
    except Exception as e:
        _eprint(f"[FEJL] Telegram exception (billede): {e}")
        log_telegram(f"EXCEPTION ved sendPhoto: {e}")
        return None


# ------------------------------------------------------------
# Dokumenter
# ------------------------------------------------------------
def send_document(doc_path, caption="", chat_id=None, silent=False, parse_mode=None):
    log_telegram(f"Sender dokument: {doc_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN")
    _chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")

    if not telegram_enabled():
        if not silent:
            _vprint(f"[TESTMODE] Ville have sendt dokument: {doc_path} (caption: {caption})", silent=False)
        log_telegram("[TESTMODE] Dokument ikke sendt – Telegram inaktiv")
        return None

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Mangler fil: {doc_path}")

        cap, pmode = _caption_and_mode(caption, parse_mode)
        if cap:
            cap = cap[:_MAX_CAPTION]

        with open(doc_path, "rb") as doc:
            files = {"document": doc}
            data = {"chat_id": _chat_id}
            if cap:
                data["caption"] = cap
            if pmode:
                data["parse_mode"] = pmode
            resp = _post_multipart(url, data=data, files=files)

        status = getattr(resp, "status_code", None)
        ok_flag = status is not None and 200 <= int(status) < 300

        if ok_flag:
            _vprint("[OK] Telegram-dokument sendt!", silent=silent)
            log_telegram("Dokument sendt OK.")
        else:
            detail = getattr(resp, "text", "")
            # Fallback ved parse-entities-fejl
            try:
                js = resp.json()
            except Exception:
                js = None
            if _is_parse_entities_error(js) and caption:
                safe_cap = _as_html_pre(caption)[:_MAX_CAPTION]
                with open(doc_path, "rb") as doc2:
                    files2 = {"document": doc2}
                    data2 = {"chat_id": _chat_id, "caption": safe_cap, "parse_mode": "HTML"}
                    resp2 = _post_multipart(url, data=data2, files=files2)
                status2 = getattr(resp2, "status_code", None)
                ok2 = status2 is not None and 200 <= int(status2) < 300
                if ok2:
                    _vprint("[OK] Telegram-dokument sendt (fallback)!", silent=silent)
                    log_telegram("Dokument sendt OK (fallback).")
                    return resp2
            _eprint(f"[FEJL] Telegram-fejl (dokument): {detail}")
            log_telegram(f"FEJL ved sendDocument: {detail}")
        return resp
    except Exception as e:
        _eprint(f"[FEJL] Telegram exception (dokument): {e}")
        log_telegram(f"EXCEPTION ved sendDocument: {e}")
        return None


# ------------------------------------------------------------
# Convenience-helpers
# ------------------------------------------------------------
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
    # Brug plain text for summary — mindre risiko for parse-fejl
    send_message(summary_text, chat_id=chat_id, parse_mode=None)
    if image_path and os.path.exists(image_path):
        send_image(image_path, caption="📈 Equity Curve", chat_id=chat_id)
    if doc_path and os.path.exists(doc_path):
        send_document(doc_path, caption="📊 Trade Journal", chat_id=chat_id)


def send_trend_graph(
    chat_id=None,
    # AUTO PATH CONVERTED
    history_path=PROJECT_ROOT / "outputs" / "performance_history.csv",
    # AUTO PATH CONVERTED
    img_path=PROJECT_ROOT / "outputs" / "balance_trend.png",
    caption="📈 Balanceudvikling",
):
    try:
        if generate_trend_graph:
            img_path = generate_trend_graph(history_path=history_path, img_path=img_path)
            if img_path and os.path.exists(img_path):
                send_image(img_path, caption=caption, chat_id=chat_id)
            else:
                send_message("Kunne ikke generere balance-trend-graf.", chat_id=chat_id)
        else:
            send_message(
                "Plot-utils ikke tilgængelig – trend-graf ikke genereret.",
                chat_id=chat_id,
            )
    except Exception as e:
        _eprint(f"[FEJL] Fejl ved trend-graf: {e}")
        log_telegram(f"EXCEPTION ved send_trend_graph: {e}")
        send_message(f"Fejl ved generering/sending af trend-graf: {e}", chat_id=chat_id)


def send_live_metrics(trades_df, balance_df, symbol="", timeframe="", thresholds=None, chat_id=None):
    """
    Send live performance-metrics og alarmer til Telegram.
    thresholds: dict, fx {"drawdown": -20, "winrate": 20, "profit": -10}
    """
    metrics = calculate_live_metrics(trades_df, balance_df)
    # Behold dine <b>-tags, men vi håndterer fallback auto i send_message
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
            alarm_msgs.append(
                f"🚨 ADVARSEL: Max drawdown under {thresholds.get('drawdown', -20)}%! ({metrics['drawdown_pct']:.2f}%)"
            )
        if check_winrate_alert(metrics, threshold=thresholds.get("winrate", 20)):
            alarm_msgs.append(
                f"🚨 ADVARSEL: Win-rate under {thresholds.get('winrate', 20)}%! ({metrics['win_rate']:.1f}%)"
            )
        if check_profit_alert(metrics, threshold=thresholds.get("profit", -10)):
            alarm_msgs.append(
                f"🚨 ADVARSEL: Profit under {thresholds.get('profit', -10)}%! ({metrics['profit_pct']:.2f}%)"
            )
    if alarm_msgs:
        for alarm in alarm_msgs:
            # Alarmer behandles som high-prio men gennem ro-på gating
            send_signal_message(alarm, symbol=symbol, dedupe_key=f"alarm|{symbol}|{alarm}", priority="high", chat_id=chat_id)
            log_telegram(alarm)


# ------------------------------------------------------------
# Manuel test
# ------------------------------------------------------------
if __name__ == "__main__":
    # Almindelig besked
    send_message("Testbesked fra din AI trading bot!")

    # High-prio gennem gating
    send_signal_message("ENTRY: BTCUSDT LONG 0.10 @ 60.00", symbol="BTCUSDT", priority="high")

    # Duplikat (bliver SUPPRESS’et under DEDUPE_TTL_SEC)
    send_signal_message("ENTRY: BTCUSDT LONG 0.10 @ 60.00", symbol="BTCUSDT", priority="high")

    # Low-prio buffer + flush
    for i in range(5):
        send_lowprio = send_signal_message  # alias via kwargs
        send_lowprio(f"FYI {i}", symbol="BTCUSDT", priority="low")
    maybe_flush_lowprio_batch()

    send_telegram_heartbeat()
    import pandas as pd

    # Dummy for live-metrics test
    balance_df = pd.DataFrame({"balance": [1000, 980, 950, 990, 970, 1005]})
    trades_df = pd.DataFrame(
        {
            "type": ["BUY", "TP", "BUY", "SL", "BUY", "TP", "SELL", "TP", "SELL", "SL"],
            "profit": [0, 0.02, 0, -0.015, 0, 0.01, 0, 0.03, 0, -0.012],
        }
    )
    send_live_metrics(
        trades_df,
        balance_df,
        symbol="BTCUSDT",
        timeframe="1h",
        thresholds={"drawdown": -2, "winrate": 60, "profit": -1},
    )
    # Evt. trend-graf
    # send_trend_graph()
