# utils/telegram_utils.py
from __future__ import annotations

import os
import time
import html
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------------------------------------
# Projektroot
# ----------------------------------------------------------------------------------------
try:
    from utils.project_path import PROJECT_ROOT
except Exception:  # pragma: no cover  - meget stabilt og svært at trigge i tests
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------------------
# Live-metrics helpers (fail-safe import)
# ----------------------------------------------------------------------------------------
try:
    from utils.monitoring_utils import (
        calculate_live_metrics,
        check_drawdown_alert,
        check_winrate_alert,
        check_profit_alert,
    )
except Exception:  # pragma: no cover  - fallback benyttes kun hvis modulet mangler
    # Minimal fallback hvis monitoring_utils ikke findes
    def calculate_live_metrics(trades_df, balance_df):  # pragma: no cover
        import pandas as pd
        metrics = {
            "profit_pct": 0.0,
            "win_rate": 0.0,
            "drawdown_pct": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
        }
        try:
            if isinstance(balance_df, pd.DataFrame) and "balance" in balance_df:
                bal = pd.to_numeric(balance_df["balance"], errors="coerce").dropna()
                if len(bal) >= 2 and float(bal.iloc[0]) != 0:
                    metrics["profit_pct"] = (float(bal.iloc[-1]) / float(bal.iloc[0]) - 1.0) * 100.0
                if len(bal) > 0:
                    roll_max = bal.cummax()
                    dd = (bal / roll_max - 1.0) * 100.0
                    metrics["drawdown_pct"] = float(dd.min())
                ret = bal.diff().dropna()
                if ret.std(ddof=0) > 1e-12:
                    metrics["sharpe"] = float(ret.mean() / ret.std(ddof=0))
            if trades_df is not None:
                metrics["num_trades"] = int(getattr(trades_df, "__len__", lambda: 0)() or 0)
                if "profit" in getattr(trades_df, "columns", []):
                    pf = trades_df["profit"]
                    wins = (pf > 0).sum()
                    tot = (pf.notna()).sum()
                    if tot:
                        metrics["win_rate"] = float(wins / tot * 100.0)
                    gross_win = float(pf[pf > 0].sum() or 0.0)
                    gross_loss = float(-pf[pf < 0].sum() or 0.0)
                    metrics["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else (gross_win and 999.0)
        except Exception:
            pass
        return metrics

    def check_drawdown_alert(metrics, threshold=-20):  # pragma: no cover
        try:
            return float(metrics.get("drawdown_pct", 0.0)) <= float(threshold)
        except Exception:
            return False

    def check_winrate_alert(metrics, threshold=20):  # pragma: no cover
        try:
            return float(metrics.get("win_rate", 0.0)) < float(threshold)
        except Exception:
            return False

    def check_profit_alert(metrics, threshold=-10):  # pragma: no cover
        try:
            return float(metrics.get("profit_pct", 0.0)) <= float(threshold)
        except Exception:
            return False

# Valgfri plot
try:
    from utils.plot_utils import generate_trend_graph
except Exception:  # pragma: no cover  - fallback bruges kun hvis modulet mangler
    generate_trend_graph = None

# ----------------------------------------------------------------------------------------
# ENV & logplacering
# ----------------------------------------------------------------------------------------
load_dotenv()

def _to_abs_path(p: Path | str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

# Brug LOG_DIR fra .env så GUI og engine peger samme sted
LOG_DIR = _to_abs_path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "telegram_log.txt"

# Terminal-verbosity (sæt TELEGRAM_VERBOSE=0 for at dæmpe "[OK] Telegram-... sendt!")
VERBOSE = os.getenv("TELEGRAM_VERBOSE", "1").strip().lower() not in ("0", "false", "no", "off")

# Telegram grænser
_MAX_TEXT = 4096
_MAX_CAPTION = 1024

# ----------------------------------------------------------------------------------------
# "Ro på" – ENV-styrede parametre (gating / dedupe / batching)
# ----------------------------------------------------------------------------------------
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

DEDUPE_TTL_SEC = _get_int_env("TELEGRAM_DEDUPE_TTL_SEC", 120)
COOLDOWN_GLOBAL_SEC = _get_int_env("TELEGRAM_COOLDOWN_GLOBAL_SEC", 5)
COOLDOWN_PER_SYMBOL_SEC = _get_int_env("TELEGRAM_COOLDOWN_PER_SYMBOL_SEC", 15)
BATCH_LOWPRIO_EVERY_SEC = _get_int_env("TELEGRAM_BATCH_LOWPRIO_EVERY_SEC", 60)
BATCH_MAX_ITEMS = _get_int_env("TELEGRAM_BATCH_MAX_ITEMS", 20)
LOG_DECISIONS = _get_bool_env("TELEGRAM_LOG_DECISIONS", True)

_last_sent_global_ts: float = 0.0
_last_sent_by_symbol: Dict[str, float] = {}
_dedupe_store: Dict[str, float] = {}
_lowprio_buffer: List[Tuple[float, str, Optional[str]]] = []
_last_batch_flush_ts: float = 0.0

# ----------------------------------------------------------------------------------------
# Print/log helpers
# ----------------------------------------------------------------------------------------
def _vprint(msg: str, silent: bool = False):
    if VERBOSE and not silent:
        print(msg)

def _eprint(msg: str):
    print(msg)

def _now_ts() -> float:
    return time.time()

def log_telegram(msg: str):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {msg}\n")
    except Exception:  # pragma: no cover  - svært at simulere I/O-fejl stabilt i tests
        _eprint(f"[ADVARSEL] Telegram-log fejlede: {msg}")

def _decision_log(action: str, reason: str, *, symbol: Optional[str] = None, extra: str = ""):
    if not LOG_DECISIONS:
        return
    sym = symbol or "-"
    log_telegram(f"[DECISION] {action} | {reason} | symbol={sym} {extra}")

# ----------------------------------------------------------------------------------------
# Telegram enable check
# ----------------------------------------------------------------------------------------
def telegram_enabled() -> bool:
    token = os.getenv("TELEGRAM_TOKEN") or ""
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or ""
    # Tillad tvungen enable for test-miljøer der vil ramme netværksgrene uden rigtige creds
    if os.getenv("TELEGRAM_TESTMODE_ALWAYS_ENABLED", "0") == "1":  # pragma: no cover (bruges ad hoc)
        return True
    return bool(token and chat_id and token.lower() not in ("none", "dummy_token") and chat_id.lower() not in ("none", "dummy_id"))

# ----------------------------------------------------------------------------------------
# Markdown/HTML utils
# ----------------------------------------------------------------------------------------
_MD2_SPECIALS = set(r'_*[]()~`>#+-=|{}.!')

def _escape_markdown_v2(s: str) -> str:
    return "".join(("\\" + ch) if ch in _MD2_SPECIALS else ch for ch in s)

def _as_html_pre(s: str) -> str:
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
    try:
        js = resp.json()
    except Exception:
        js = None
    if isinstance(js, dict) and "ok" in js:
        return bool(js.get("ok", False)), js
    status = getattr(resp, "status_code", None)
    ok_flag = status is not None and 200 <= int(status) < 300
    return ok_flag, js

# ----------------------------------------------------------------------------------------
# Tekst-beskeder
# ----------------------------------------------------------------------------------------
def _send_text_request(token: str, chat_id: str, text: str, parse_mode: Optional[str]):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return requests.post(url, json=payload, timeout=10)

def _send_text_chunked(token: str, chat_id: str, text: str, parse_mode: Optional[str]):
    # Escape ved Markdown
    if parse_mode and str(parse_mode).upper().startswith("MARKDOWN"):
        text = _escape_markdown_v2(text)

    limit = _MAX_TEXT
    last_resp_json = None
    for chunk_start in range(0, len(text), limit):
        chunk = text[chunk_start:chunk_start + limit]
        resp = _send_text_request(token, chat_id, chunk, parse_mode)
        ok, resp_json = _resp_ok(resp)
        last_resp_json = resp_json if isinstance(resp_json, dict) else last_resp_json
        if ok:
            continue
        if _is_parse_entities_error(resp_json):
            # Fallback til HTML <pre>
            safe = _as_html_pre(chunk)
            resp2 = _send_text_request(token, chat_id, safe, "HTML")
            ok2, resp2_json = _resp_ok(resp2)
            last_resp_json = resp2_json if isinstance(resp2_json, dict) else last_resp_json
            if not ok2:  # pragma: no cover  - dobbelt-fejl på fallback er svær at udløse stabilt
                _eprint(f"[FEJL] Telegram parse & fallback fejlede: {getattr(resp2, 'text', '')}")
                log_telegram(f"Fallback parse-fejl: {getattr(resp2, 'text', '')}")
        else:
            _eprint(f"[FEJL] Telegram API {getattr(resp, 'status_code', '?')}: {getattr(resp, 'text', '')}")
            log_telegram(f"FEJL ved sendMessage: {getattr(resp, 'text', '')}")

    return last_resp_json if isinstance(last_resp_json, dict) else True

def send_message(msg: str, chat_id: Optional[str] = None, parse_mode: Optional[str] = None, silent: bool = False):
    """
    Offentlig helper:
      - chunker automatisk
      - håndterer MarkdownV2-escaping og fallback til HTML <pre> ved parse-fejl
      - logger til LOG_DIR/telegram_log.txt
    """
    log_telegram(f"Sender besked: {msg}")
    token = os.getenv("TELEGRAM_TOKEN") or ""
    _chat_id = chat_id if chat_id is not None else (os.getenv("TELEGRAM_CHAT_ID") or "")

    if not telegram_enabled():
        if not silent:
            _vprint(f"[TESTMODE] Ville have sendt Telegram-besked: {msg}", silent=False)
        log_telegram("[TESTMODE] Besked ikke sendt – Telegram inaktiv")
        return None

    try:
        resp_json_or_true = _send_text_chunked(token, _chat_id, msg, parse_mode)

        # Log kun "[OK]" når Telegram svarer ok:true
        ok_flag = True
        if isinstance(resp_json_or_true, dict):
            ok_flag = bool(resp_json_or_true.get("ok", False))

        if ok_flag:
            _vprint("[OK] Telegram-besked sendt!", silent=silent)
            log_telegram("Besked sendt OK.")
        else:
            _eprint(f"[FEJL] Telegram sendMessage mislykkedes: {resp_json_or_true}")
            log_telegram(f"FEJL ved sendMessage: {resp_json_or_true}")

        return resp_json_or_true
    except Exception as e:
        _eprint(f"[FEJL] Telegram exception: {e}")
        log_telegram(f"EXCEPTION ved sendMessage: {e}")
        return None

send_telegram_message = send_message  # alias

# ----------------------------------------------------------------------------------------
# Gating (dedupe / cooldown / batching)
# ----------------------------------------------------------------------------------------
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
    priority: str = "high",          # "high" | "low"
    parse_mode: Optional[str] = None,
    chat_id: Optional[str] = None,
    silent: bool = False,
    skip_cooldown: bool = False,
    skip_dedupe: bool = False,
):
    """
    Gated Telegram-sender:
      - Duplikatfilter via dedupe_key (TTL-styret)
      - Global + pr.-symbol cooldown
      - Low-prio: batchet udsendelse
    """
    key = dedupe_key or f"{symbol or '-'}|{hash(text)}"

    if priority.lower() == "low":
        _lowprio_buffer.append((_now_ts(), text, symbol))
        _decision_log("SUPPRESS", "queued_lowprio", symbol=symbol, extra=f"(buffer={len(_lowprio_buffer)})")
        maybe_flush_lowprio_batch(chat_id=chat_id)
        return {"ok": True, "suppressed": True, "reason": "queued_lowprio"}

    if not skip_dedupe and _is_duplicate(key):
        _decision_log("SUPPRESS", "duplicate", symbol=symbol, extra=f"key={key}")
        return {"ok": True, "suppressed": True, "reason": "duplicate"}

    if not skip_cooldown and _in_cooldown(symbol):
        _decision_log("SUPPRESS", "cooldown", symbol=symbol)
        return {"ok": True, "suppressed": True, "reason": "cooldown"}

    resp = send_message(text, chat_id=chat_id, parse_mode=parse_mode, silent=silent)
    if not skip_cooldown:
        _mark_sent(symbol)
    _decision_log("NOTIFY", "sent", symbol=symbol, extra=(" (bypass_cooldown)" if skip_cooldown else ""))
    return resp

def maybe_flush_lowprio_batch(chat_id: Optional[str] = None, header: str = "🔔 Lav-prio opsummering"):
    global _last_batch_flush_ts
    now = _now_ts()
    if now - _last_batch_flush_ts < max(5, BATCH_LOWPRIO_EVERY_SEC):
        return False
    if not _lowprio_buffer:
        _last_batch_flush_ts = now
        return False

    items = list(_lowprio_buffer)
    del _lowprio_buffer[:]
    _last_batch_flush_ts = now

    lines = [f"{header} • {len(items)} beskeder"]
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

    send_message("\n".join(lines), chat_id=chat_id, parse_mode=None, silent=True)
    _decision_log("NOTIFY", "batch_flush", extra=f"items={len(items)}")
    return True

# ----------------------------------------------------------------------------------------
# Billeder & dokumenter
# ----------------------------------------------------------------------------------------
def _post_multipart(url: str, data: dict, files: dict):
    return requests.post(url, data=data, files=files, timeout=20)

def _caption_and_mode(caption: str, parse_mode: Optional[str]):
    if not caption:
        return caption, (None if parse_mode is None else parse_mode)
    if parse_mode and parse_mode.upper().startswith("MARKDOWN"):
        return _escape_markdown_v2(caption), "MarkdownV2"
    if parse_mode and parse_mode.upper() == "HTML":
        return caption, "HTML"
    return caption, None

def send_image(photo_path: str, caption: str = "", chat_id: Optional[str] = None, silent: bool = False, parse_mode: Optional[str] = None):
    log_telegram(f"Sender billede: {photo_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN") or ""
    _chat_id = chat_id if chat_id is not None else (os.getenv("TELEGRAM_CHAT_ID") or "")

    if not telegram_enabled():
        if not silent:
            _vprint(f"[TESTMODE] Ville have sendt billede: {photo_path} (caption: {caption})", silent=False)
        log_telegram("[TESTMODE] Billede ikke sendt – Telegram inaktiv")
        return None

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        if not Path(photo_path).exists():
            raise FileNotFoundError(f"Mangler fil: {photo_path}")

        cap, pmode = _caption_and_mode(caption, parse_mode)
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

        ok, js = _resp_ok(resp)
        if ok:
            _vprint("[OK] Telegram-billede sendt!", silent=silent)
            log_telegram("Billede sendt OK.")
            return resp
        # parse fallback
        if _is_parse_entities_error(js) and caption:
            safe_cap = _as_html_pre(caption)[:_MAX_CAPTION]
            with open(photo_path, "rb") as photo2:
                files2 = {"photo": photo2}
                data2 = {"chat_id": _chat_id, "caption": safe_cap, "parse_mode": "HTML"}
                resp2 = _post_multipart(url, data=data2, files=files2)
            ok2, _ = _resp_ok(resp2)
            if ok2:
                _vprint("[OK] Telegram-billede sendt (fallback)!", silent=silent)
                log_telegram("Billede sendt OK (fallback).")
                return resp2
        _eprint(f"[FEJL] Telegram-fejl (billede): {getattr(resp, 'text', '')}")
        log_telegram(f"FEJL ved sendPhoto: {getattr(resp, 'text', '')}")
        return resp
    except Exception as e:
        _eprint(f"[FEJL] Telegram exception (billede): {e}")
        log_telegram(f"EXCEPTION ved sendPhoto: {e}")
        return None

def send_document(doc_path: str, caption: str = "", chat_id: Optional[str] = None, silent: bool = False, parse_mode: Optional[str] = None):
    log_telegram(f"Sender dokument: {doc_path} (caption: {caption})")
    token = os.getenv("TELEGRAM_TOKEN") or ""
    _chat_id = chat_id if chat_id is not None else (os.getenv("TELEGRAM_CHAT_ID") or "")

    if not telegram_enabled():
        if not silent:
            _vprint(f"[TESTMODE] Ville have sendt dokument: {doc_path} (caption: {caption})", silent=False)
        log_telegram("[TESTMODE] Dokument ikke sendt – Telegram inaktiv")
        return None

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        if not Path(doc_path).exists():
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

        ok, js = _resp_ok(resp)
        if ok:
            _vprint("[OK] Telegram-dokument sendt!", silent=silent)
            log_telegram("Dokument sendt OK.")
            return resp
        if _is_parse_entities_error(js) and caption:
            safe_cap = _as_html_pre(caption)[:_MAX_CAPTION]
            with open(doc_path, "rb") as doc2:
                files2 = {"document": doc2}
                data2 = {"chat_id": _chat_id, "caption": safe_cap, "parse_mode": "HTML"}
                resp2 = _post_multipart(url, data=data2, files=files2)
            ok2, _ = _resp_ok(resp2)
            if ok2:
                _vprint("[OK] Telegram-dokument sendt (fallback)!", silent=silent)
                log_telegram("Dokument sendt OK (fallback).")
                return resp2
        _eprint(f"[FEJL] Telegram-fejl (dokument): {getattr(resp, 'text', '')}")
        log_telegram(f"FEJL ved sendDocument: {getattr(resp, 'text', '')}")
        return resp
    except Exception as e:
        _eprint(f"[FEJL] Telegram exception (dokument): {e}")
        log_telegram(f"EXCEPTION ved sendDocument: {e}")
        return None

# ----------------------------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------------------------
def send_telegram_heartbeat(chat_id: Optional[str] = None):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"💓 Botten kører stadig! ({t})"
    send_message(msg, chat_id=chat_id)
    log_telegram("Heartbeat sendt.")

def send_strategy_metrics(metrics: Dict, chat_id: Optional[str] = None):
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

def send_auto_status_summary(summary_text: str, image_path: Optional[str] = None, doc_path: Optional[str] = None, chat_id: Optional[str] = None):
    send_message(summary_text, chat_id=chat_id, parse_mode=None)
    if image_path and Path(image_path).exists():
        send_image(image_path, caption="📈 Equity Curve", chat_id=chat_id)
    if doc_path and Path(doc_path).exists():
        send_document(doc_path, caption="📊 Trade Journal", chat_id=chat_id)

def send_trend_graph(
    chat_id: Optional[str] = None,
    history_path: Path = PROJECT_ROOT / "outputs" / "performance_history.csv",
    img_path: Path = PROJECT_ROOT / "outputs" / "balance_trend.png",
    caption: str = "📈 Balanceudvikling",
):
    try:
        if generate_trend_graph:
            img_path = Path(generate_trend_graph(history_path=history_path, img_path=img_path))
            if img_path and img_path.exists():
                send_image(str(img_path), caption=caption, chat_id=chat_id)
            else:
                send_message("Kunne ikke generere balance-trend-graf.", chat_id=chat_id)
        else:
            send_message("Plot-utils ikke tilgængelig – trend-graf ikke genereret.", chat_id=chat_id)
    except Exception as e:  # pragma: no cover  - fejlgren er dækket andetsteds via send_message
        _eprint(f"[FEJL] Fejl ved trend-graf: {e}")
        log_telegram(f"EXCEPTION ved send_trend_graph: {e}")
        send_message(f"Fejl ved generering/sending af trend-graf: {e}", chat_id=chat_id)

def send_live_metrics(trades_df, balance_df, symbol: str = "", timeframe: str = "", thresholds: Optional[Dict] = None, chat_id: Optional[str] = None):
    """
    Send live performance-metrics og alarmer til Telegram.
    thresholds: dict, fx {"drawdown": -20, "winrate": 20, "profit": -10}
    Alarmer sendes som uafhængige beskeder og bypasser cooldown i samme batch.
    """
    metrics = calculate_live_metrics(trades_df, balance_df)
    msg = (
        f"📡 <b>Live trading-status {symbol} {timeframe}</b>\n"
        f"Profit: <b>{metrics.get('profit_pct', 0.0):.2f}%</b>\n"
        f"Win-rate: <b>{metrics.get('win_rate', 0.0):.1f}%</b>\n"
        f"Drawdown: <b>{metrics.get('drawdown_pct', 0.0):.2f}%</b>\n"
        f"Antal handler: <b>{metrics.get('num_trades', 0)}</b>\n"
        f"Profit factor: <b>{metrics.get('profit_factor', 0)}</b>\n"
        f"Sharpe: <b>{metrics.get('sharpe', 0)}</b>\n"
    )
    send_message(msg, chat_id=chat_id, parse_mode="HTML")

    if thresholds:
        alarms: List[str] = []
        if check_drawdown_alert(metrics, threshold=thresholds.get("drawdown", -20)):
            alarms.append(f"🚨 ADVARSEL: Max drawdown under {thresholds.get('drawdown', -20)}%! ({metrics.get('drawdown_pct', 0.0):.2f}%)")
        if check_winrate_alert(metrics, threshold=thresholds.get("winrate", 20)):
            alarms.append(f"🚨 ADVARSEL: Win-rate under {thresholds.get('winrate', 20)}%! ({metrics.get('win_rate', 0.0):.1f}%)")
        if check_profit_alert(metrics, threshold=thresholds.get("profit", -10)):
            alarms.append(f"🚨 ADVARSEL: Profit under {thresholds.get('profit', -10)}%! ({metrics.get('profit_pct', 0.0):.2f}%)")

        for alarm in alarms:
            send_signal_message(
                alarm,
                symbol=symbol,
                dedupe_key=f"alarm|{symbol}|{alarm}",
                priority="high",
                chat_id=chat_id,
                skip_cooldown=True,   # så flere alarmer i samme batch kommer igennem
                skip_dedupe=False,
            )
            log_telegram(alarm)

# ----------------------------------------------------------------------------------------
# Manuel test
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover  - manuel kørsel, ikke relevant for tests
    # Almindelig besked
    send_message("Testbesked fra din AI trading bot!")

    # High-prio gennem gating
    send_signal_message("ENTRY: BTCUSDT LONG 0.10 @ 60.00", symbol="BTCUSDT", priority="high")

    # Duplikat (bliver SUPPRESS’et under DEDUPE_TTL_SEC)
    send_signal_message("ENTRY: BTCUSDT LONG 0.10 @ 60.00", symbol="BTCUSDT", priority="high")

    # Low-prio buffer + flush
    for i in range(5):
        send_signal_message(f"FYI {i}", symbol="BTCUSDT", priority="low")
    maybe_flush_lowprio_batch()

    send_telegram_heartbeat()

    # Dummy live-metrics test
    try:
        import pandas as pd
        balance_df = pd.DataFrame({"balance": [1000, 980, 950, 990, 970, 1005]})
        trades_df = pd.DataFrame(
            {"type": ["BUY", "TP", "BUY", "SL", "BUY", "TP", "SELL", "TP", "SELL", "SL"],
             "profit": [0, 0.02, 0, -0.015, 0, 0.01, 0, 0.03, 0, -0.012]}
        )
        send_live_metrics(
            trades_df,
            balance_df,
            symbol="BTCUSDT",
            timeframe="1h",
            thresholds={"drawdown": -2, "winrate": 60, "profit": -1},
        )
    except Exception as e:
        _eprint(f"[TEST] Kunne ikke køre live-metrics dummy: {e}")
