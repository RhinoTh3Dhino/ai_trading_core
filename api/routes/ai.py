from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from llm.anthropic_client import ask_claude
from utils.project_path import PROJECT_ROOT

router = APIRouter(prefix="/ai", tags=["ai"])

ROOT = Path(PROJECT_ROOT)
LOGS = ROOT / "logs"
API_DIR = ROOT / "api"


# ---------------------------
# Hjælpere (robuste IO + utils)
# ---------------------------
def _read_csv(
    path: Path, usecols: Optional[List[str]] = None, tail: Optional[int] = None
) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if usecols:
            keep = [c for c in usecols if c in df.columns]
            df = df[keep] if keep else df
        if tail:
            df = df.tail(int(tail))
        return df
    except Exception:
        return pd.DataFrame()


def _load_json_list(path: Path, tail: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data[-tail:] if tail else data
    except Exception:
        return []


def _has_llm() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _call_claude(system: str, user: str) -> Optional[str]:
    try:
        return ask_claude(system, user)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


# ---------------------------
# /ai/summary
# ---------------------------
@router.get("/summary")
def ai_summary(
    limit_days: int = Query(7, alias="n_days", ge=1, le=60),
    signal_limit: int = Query(12, ge=1, le=200),
):
    """
    Giver en kort AI-sammendrag af performance og signaler.
    - limit_days: antal dage fra daily_metrics
    - signal_limit: antal seneste signal-objekter fra api/sim_signals.json
    """
    # Saml kompakt “kontekst”
    daily = _read_csv(LOGS / "daily_metrics.csv")
    equity = _read_csv(LOGS / "equity.csv", usecols=["date", "equity"], tail=90)
    recent_signals = _load_json_list(API_DIR / "sim_signals.json", tail=signal_limit)

    payload = {
        "daily_metrics_last_days": daily.tail(limit_days).to_dict(orient="records"),
        "equity_last_90": equity.to_dict(orient="records"),
        "recent_signals": recent_signals,
    }

    # Hvis LLM er tilgængelig -> spørg
    if _has_llm():
        system = (
            "Du er en nøgtern trading-assistent. "
            "Svar kort, klart og handlingsorienteret. Ingen hype."
        )
        user = (
            "Lav en 6–8 punkters status for seneste periode og slut med 2 konkrete NEXT STEPS. "
            "Opsummer PnL/medvind/modvind, risiko og eventuelle regime-skift. "
            "Data (JSON):\n\n" + json.dumps(payload)[:12000]
        )
        text = _call_claude(system, user)
        if text:
            return {"text": text}

    # Fallback uden LLM: lav et kompakt heuristisk resume
    dm = daily.tail(limit_days).copy() if not daily.empty else pd.DataFrame()
    net = _safe_float(dm["net_pnl"].sum()) if "net_pnl" in dm.columns else 0.0
    trades = int(dm["trades"].sum()) if "trades" in dm.columns else 0
    wr = _safe_float(dm["win_rate"].mean()) if "win_rate" in dm.columns else 0.0
    max_dd = (
        _safe_float(dm["max_dd"].min()) if "max_dd" in dm.columns and len(dm) else 0.0
    )

    eq = equity.copy()
    eq_chg = 0.0
    if not eq.empty and "equity" in eq.columns:
        try:
            start = _safe_float(eq["equity"].iloc[0], 0.0)
            end = _safe_float(eq["equity"].iloc[-1], 0.0)
            if start > 0:
                eq_chg = (end - start) / start * 100.0
        except Exception:
            pass

    bullets = [
        f"Net PnL (≈{limit_days} dage): {net:.2f}",
        f"Handler: {trades}",
        f"Gns. win-rate: {_fmt_pct(wr)}",
        f"Max drawdown (min): {_fmt_pct(max_dd)}",
        f"Equity-ændring (sidste ~90 punkter): {_fmt_pct(eq_chg)}",
        f"Seneste {min(len(recent_signals), signal_limit)} signaler læst ind.",
    ]
    steps = [
        "Hæv/lav ATR-filter, og re-tjek hit-rate.",
        "Kør walk-forward backtest på 3–6m data for robusthed.",
    ]
    text_fb = "• " + "\n• ".join(bullets) + "\n\nNEXT STEPS:\n- " + "\n- ".join(steps)
    return {"text": text_fb}


# ---------------------------
# /ai/explain_trade
# ---------------------------
@router.get("/explain_trade")
def ai_explain_trade(
    i: int = Query(0, ge=0, description="0 = seneste handel, 1 = næstseneste, osv."),
    context_bars: int = Query(
        60, ge=10, le=500, description="Antal seneste signaler/kontext-punkter"
    ),
):
    """
    Forklar én handel baseret på seneste fills + signaler.
    - i: hvilken handel (0 = seneste)
    - context_bars: hvor mange seneste signalpunkter der sendes som kontekst
    """
    fills = _read_csv(
        LOGS / "fills.csv",
        # prøv at få mest relevante kolonner med, men vær tolerant
        usecols=None,
        tail=None,
    )

    # Normalisér/uddyb fills til kendte felter
    if not fills.empty:
        # lav alt lower-case for robust rename
        fills = fills.rename(columns={c: c.lower() for c in fills.columns})
        # aliaser
        alias = {
            "timestamp": "ts",
            "time": "ts",
            "datetime": "ts",
            "quantity": "qty",
            "pnl": "pnl_realized",
        }
        for s, d in alias.items():
            if s in fills.columns and d not in fills.columns:
                fills = fills.rename(columns={s: d})

        # sortér (seneste først hvis 'ts' findes)
        if "ts" in fills.columns:
            with pd.option_context("mode.chained_assignment", None):
                fills["ts"] = pd.to_datetime(fills["ts"], errors="coerce")
            fills = fills.sort_values("ts", ascending=True).reset_index(drop=True)

    # vælg målhandel (seneste = i=0)
    chosen: Optional[Dict[str, Any]] = None
    if not fills.empty:
        idx = len(fills) - 1 - int(i)
        if 0 <= idx < len(fills):
            chosen = fills.iloc[idx].to_dict()

    # Hent kontekst: seneste signaler og equity
    signals = _load_json_list(API_DIR / "sim_signals.json", tail=context_bars)
    equity = _read_csv(LOGS / "equity.csv", usecols=["date", "equity"], tail=120)
    daily = _read_csv(LOGS / "daily_metrics.csv", tail=14)

    # Hvis LLM er til rådighed, spørg om en forklaring
    if _has_llm():
        system = (
            "Du er en nøgtern trading-assistent. Forklar kort hvorfor handlen gav mening ud fra kontekst, "
            "risiko/position sizing og hvad man skal holde øje med fremadrettet. "
            "Skriv maks 150 ord i 3–5 punktopstillinger."
        )
        if chosen:
            user = (
                "Forklar denne handel ud fra konteksten nedenfor.\n\n"
                f"HANDEL (JSON): {json.dumps(chosen, default=str)[:2000]}\n\n"
                f"SENESTE SIGNALER (JSON, {len(signals)} stk): {json.dumps(signals, default=str)[:6000]}\n\n"
                f"EQUITY (sidste {len(equity)} punkter): {json.dumps(equity.to_dict(orient='records'))[:4000]}\n\n"
                f"DAILY METRICS (seneste {len(daily)}): {json.dumps(daily.to_dict(orient='records'))[:4000]}\n\n"
                "Giv en kort, konkret forklaring i punktopstilling."
            )
        else:
            # fallback: ingen fills endnu – forklar seneste signal
            user = (
                "Ingen fills tilgængelige. Forklar i stedet SENESTE SIGNAL ud fra konteksten.\n\n"
                f"SENESTE SIGNALER (JSON, {len(signals)} stk): {json.dumps(signals, default=str)[:6000]}\n\n"
                f"EQUITY (sidste {len(equity)} punkter): {json.dumps(equity.to_dict(orient='records'))[:4000]}\n\n"
                "Giv en kort, konkret forklaring i punktopstilling."
            )
        text = _call_claude(system, user)
        if text:
            return {"text": text}

    # Heuristisk fallback-tekst uden LLM
    if chosen:
        sym = str(chosen.get("symbol", "N/A"))
        side = str(chosen.get("side", "N/A")).upper()
        qty = _safe_float(chosen.get("qty", chosen.get("quantity", 0.0)))
        px = _safe_float(chosen.get("price", 0.0))
        pnl = _safe_float(chosen.get("pnl_realized", 0.0))
        com = _safe_float(chosen.get("commission", 0.0))
        bullets = [
            f"Handel: {side} {qty:g} {sym} @ {px:.2f}",
            f"Realiseret PnL: {pnl:.2f}  (kommission: {com:.2f})",
            f"Kontekst: {len(signals)} seneste signaler indlæst; equity-punkter: {len(equity)}.",
            "Tolkning: Brug signalets 'confidence' og regime til at validere entries/exits.",
            "Fremad: Overvej ATR-filter/cooldown for at reducere noise og slippage.",
        ]
        return {"text": "• " + "\n• ".join(bullets)}

    # Ingen fills og ingen signaler: sidste fallback
    return {
        "text": (
            "Ingen handler registreret endnu og ingen signal-kontekst tilgængelig. "
            "Vent på første signal/fill, eller tjek at live-daemonen kører og skriver til api/sim_signals.json og logs/fills.csv."
        )
    }
