# utils/alerts_eval.py
"""
Send kort/lang Telegram-alert baseret på seneste Claude-evaluering.

Funktioner:
- load_eval: indlæs eval-JSON fra outputs/evals/last_eval.json (standard)
- format_eval_summary: kort 3-linjers status til Telegram
- format_eval_markdown: udvidet tekst inkl. top-risici og anbefalinger
- send_eval_alert: send kort eller lang besked (+ valgfrit vedhæftet JSON)
- CLI: python -m utils.alerts_eval --path outputs/evals/last_eval.json --long --document

Afhænger valgfrit af utils.telegram_utils:
- send_message(text: str, parse_mode: Optional[str], disable_web_page_preview: bool = True)
- send_document(file_path: str, caption: Optional[str] = None)

Hvis modulet ikke findes, degraderer vi til stdout (print) for smoke-tests/CI.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Valgfri Telegram-integration (degraderer til print hvis ikke tilgængelig) ---
_TG_SEND_MESSAGE = None
_TG_SEND_DOCUMENT = None
try:
    # Typisk eksisterer dette som utils/telegram_utils.py i dit repo
    from utils.telegram_utils import send_message as _TG_SEND_MESSAGE  # type: ignore

    try:
        from utils.telegram_utils import send_document as _TG_SEND_DOCUMENT  # type: ignore
    except Exception:
        _TG_SEND_DOCUMENT = None
except Exception:
    # Ingen Telegram i miljøet – vi kører bare med stdout-fallback
    _TG_SEND_MESSAGE = None
    _TG_SEND_DOCUMENT = None


# --------------------------- Hjælpestrukturer ---------------------------


@dataclass
class EvalCore:
    run_id: str
    model: str
    edge_score: Optional[int]
    actionability: Optional[str]
    confidence: Optional[float]
    trades: Optional[int]
    win_rate: Optional[float]
    avg_rr: Optional[float]
    pnl_sum: Optional[float]
    max_dd_est: Optional[float]
    risk_flags: List[str]
    anomalies: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


# --------------------------- Utility-funktioner ---------------------------


def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Hent nested værdi med 'a.b.c' sti – returnér default hvis ikke fundet."""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _status_emoji(edge_score: Optional[int], actionability: Optional[str]) -> str:
    """Map edge_score/actionability til et kort status-emoji."""
    if edge_score is None:
        return "❔"
    if edge_score >= 80:
        return "🟢"
    if edge_score >= 60:
        return "🟡"
    if edge_score >= 40:
        return "🟠"
    return "🔴"


def _sev_emoji(sev: str) -> str:
    sev = (sev or "").upper()
    return {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}.get(sev, "❔")


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "–"


def _fmt_float(x: Optional[float], nd: int = 2) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "–"


def _try_import_schema() -> Optional[Any]:
    """
    Forsøger at importere Pydantic-skemaet (EdgeEvalV1).
    Hvis ikke tilgængeligt, returnér None og kør med let validering.
    """
    try:
        from evaluation.schemas import EdgeEvalV1  # type: ignore

        return EdgeEvalV1
    except Exception:
        return None


# --------------------------- Indlæsning og let validering ---------------------------


def load_eval(
    path: str | Path = "outputs/evals/last_eval.json",
) -> Tuple[Dict[str, Any], EvalCore]:
    """
    Indlæs eval-JSON og konverter til EvalCore.
    Validerer mod Pydantic-skema hvis tilgængeligt, ellers minimal key-checks.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval JSON ikke fundet: {p}")

    raw_txt = p.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(raw_txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ugyldig JSON i {p}: {e}") from e

    # Hvis Pydantic-skema er tilgængeligt, valider først
    EdgeEvalV1 = _try_import_schema()
    if EdgeEvalV1 is not None:
        try:
            obj = EdgeEvalV1.model_validate(data)
            data = json.loads(obj.model_dump_json(by_alias=True))
        except Exception as e:
            # Falder tilbage til let validering
            pass

    # Minimal ekstraktion (robust mod manglende felter)
    core = EvalCore(
        run_id=_safe_get(data, "run_id", ""),
        model=_safe_get(data, "model", ""),
        edge_score=_safe_get(data, "edge_score"),
        actionability=_safe_get(data, "actionability"),
        confidence=_safe_get(data, "confidence"),
        trades=_safe_get(data, "input_stats.trades"),
        win_rate=_safe_get(data, "metrics.win_rate"),
        avg_rr=_safe_get(data, "metrics.avg_rr"),
        pnl_sum=_safe_get(data, "metrics.pnl_sum"),
        max_dd_est=_safe_get(data, "metrics.max_dd_est"),
        risk_flags=_safe_get(data, "risk_flags", []) or [],
        anomalies=_safe_get(data, "anomalies", []) or [],
        recommendations=_safe_get(data, "recommendations", []) or [],
    )
    return data, core


# --------------------------- Formatér beskeder ---------------------------


def format_eval_summary(core: EvalCore) -> str:
    """
    Kort 3–5 linjers besked til Telegram. Designet til at passe i ét skærmbillede.
    """
    emoji = _status_emoji(core.edge_score, core.actionability)
    flags = ", ".join(core.risk_flags[:2]) if core.risk_flags else "–"
    return "\n".join(
        [
            f"Claude Eval {emoji}",
            f"Edge: {core.edge_score if core.edge_score is not None else '–'} | {core.actionability or '–'} | conf={_fmt_float(core.confidence, 2)}",
            f"Trades={core.trades or 0} | WinRate={_fmt_pct(core.win_rate)} | RR={_fmt_float(core.avg_rr, 2)} | PnL={_fmt_float(core.pnl_sum, 2)} | DD={_fmt_float(core.max_dd_est, 2)}",
            f"Risici: {flags}",
            "— (uddannelse/udvikling – ikke finansiel rådgivning) —",
        ]
    )


def format_eval_markdown(
    data: Dict[str, Any], core: EvalCore, max_recs: int = 3, max_anoms: int = 3
) -> str:
    """
    Længere beskrivelse (Markdown) til Telegram (eller intern log).
    Viser top-risici, anomalier og anbefalinger.
    """
    lines: List[str] = []
    lines.append(f"*Claude Eval {_status_emoji(core.edge_score, core.actionability)}*")
    lines.append(
        f"*Edge:* {core.edge_score if core.edge_score is not None else '–'} | *Action:* {core.actionability or '–'} | *Conf:* {_fmt_float(core.confidence, 2)}"
    )
    lines.append(
        f"*Trades:* {core.trades or 0} | *WinRate:* {_fmt_pct(core.win_rate)} | *RR:* {_fmt_float(core.avg_rr)} | *PnL:* {_fmt_float(core.pnl_sum)} | *DD:* {_fmt_float(core.max_dd_est)}"
    )

    # Risici
    if core.risk_flags:
        lines.append("\n*Top-risici:*")
        for rf in core.risk_flags[:3]:
            lines.append(f"• {rf}")

    # Anomalier
    if core.anomalies:
        lines.append("\n*Anomalier:*")
        for a in core.anomalies[:max_anoms]:
            sev = _sev_emoji(str(a.get("severity", "")))
            typ = a.get("type", "unknown")
            msg = a.get("message", "")
            lines.append(f"• {sev} *{typ}*: {msg}")

    # Anbefalinger
    if core.recommendations:
        lines.append("\n*Anbefalinger:*")
        for r in core.recommendations[:max_recs]:
            param = r.get("param", "param")
            cur = r.get("current", "–")
            prop = r.get("proposed", "–")
            why = r.get("rationale", "")
            lines.append(f"• *{param}*: {cur} → {prop} — {why}")

    # No-action reason (hvis angivet)
    nar = str(_safe_get(data, "no_action_reason", "") or "").strip()
    if nar:
        lines.append(f"\n*Bemærkning:* {nar}")

    lines.append("\n_(uddannelse/udvikling – ikke finansiel rådgivning)_")
    return "\n".join(lines)


# --------------------------- Afsendelse ---------------------------


def _send_text(text: str, parse_mode: Optional[str] = None) -> None:
    """Wrapper: Telegram hvis muligt, ellers stdout."""
    if _TG_SEND_MESSAGE:
        # parse_mode kan være "Markdown" eller "HTML" afhængigt af dit utils
        _TG_SEND_MESSAGE(text, parse_mode=parse_mode, disable_web_page_preview=True)  # type: ignore
    else:
        print("--- TELEGRAM (FAKE) ---")
        print(text)
        print("-----------------------")


def _send_file(file_path: str, caption: Optional[str] = None) -> None:
    """Wrapper for dokumenter (JSON-vedhæftning)."""
    if _TG_SEND_DOCUMENT:
        _TG_SEND_DOCUMENT(file_path, caption=caption)  # type: ignore
    else:
        print(f"--- TELEGRAM FILE (FAKE) --- {file_path}")
        if caption:
            print(f"CAPTION: {caption}")
        print("---------------------------")


def send_eval_alert(
    path: str | Path = "outputs/evals/last_eval.json",
    long: bool = False,
    also_send_document: bool = False,
) -> bool:
    """
    Send eval-alert til Telegram.
    - long=False: kort 3–5 linjer
    - long=True: markdown med top-risici, anomalier og anbefalinger
    - also_send_document=True: vedhæft eval JSON som dokument

    Returnerer True hvis OK, False ved fejl (og sender fejltekst).
    """
    try:
        data, core = load_eval(path)
    except Exception as e:
        _send_text(f"Claude Eval ❌ Kunne ikke indlæse eval: {e}")
        return False

    try:
        if long:
            msg = format_eval_markdown(data, core)
            _send_text(msg, parse_mode="Markdown")
        else:
            msg = format_eval_summary(core)
            _send_text(msg, parse_mode=None)

        if also_send_document:
            p = str(Path(path))
            _send_file(p, caption=f"Eval JSON • run_id={core.run_id or 'unknown'}")

        return True
    except Exception as e:
        _send_text(f"Claude Eval ❌ Fejl under afsendelse: {e}")
        return False


# --------------------------- CLI ---------------------------


def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    """Minimal argparse uden afhængigheder – egnet til simple CLI-kald."""
    args: Dict[str, Any] = {
        "path": "outputs/evals/last_eval.json",
        "long": False,
        "document": False,
    }
    it = iter(argv)
    for token in it:
        if token in ("-p", "--path"):
            args["path"] = next(it, args["path"])
        elif token in ("-l", "--long"):
            args["long"] = True
        elif token in ("-d", "--document"):
            args["document"] = True
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """
    Eksempler:
      python -m utils.alerts_eval
      python -m utils.alerts_eval --path outputs/evals/last_eval.json --long
      python -m utils.alerts_eval -p outputs/evals/last_eval.json -l -d
    """
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_argv(argv)
    ok = send_eval_alert(
        path=args["path"],
        long=args["long"],
        also_send_document=args["document"],
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
