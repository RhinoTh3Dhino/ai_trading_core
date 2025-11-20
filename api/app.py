# api/app.py
from __future__ import annotations

import contextlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ⬇️ NYT: Prometheus-klient til /metrics
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from pydantic import BaseModel

# ---------------------------
# Paths & App
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
API_DIR = ROOT / "api"
LAST_TS_FILE = LOGS / ".last_bar_ts"

app = FastAPI(title="AI Trading Bot API (Paper Demo)", version="0.2.3")

# ---------------------------
# CORS (kan udvides via ENV)
# ---------------------------
_default_origins = {"http://localhost:8501", "http://127.0.0.1:8501"}
extra = os.getenv("EXTRA_CORS_ORIGINS", "")
if extra:
    _default_origins |= {x.strip() for x in extra.split(",") if x.strip()}

if os.getenv("WIDE_CORS", "").lower() in ("1", "true", "yes"):
    allow_origins = ["*"]
else:
    allow_origins = sorted(_default_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Globalt: sikre UTF-8 for JSON
# ---------------------------
@app.middleware("http")
async def ensure_utf8_json(request, call_next):
    """
    Sørger for at alle JSON-svar har 'charset=utf-8' i Content-Type.
    Løser forkert visning af bullets/æøå i Windows-klienter (fx Invoke-RestMethod).
    """
    resp = await call_next(request)
    ct = resp.headers.get("content-type", "")
    if ct.startswith("application/json") and "charset" not in ct.lower():
        resp.headers["content-type"] = "application/json; charset=utf-8"
    return resp


# ---------------------------
# Pydantic models (bruges hvor format er fast)
# ---------------------------
class EquityPoint(BaseModel):
    date: str
    equity: float


class DailyMetric(BaseModel):
    date: str
    signal_count: int
    trades: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    max_dd: float
    sharpe_d: float


# ---------------------------
# Prometheus metrics (NYT)
# ---------------------------
_METRICS: CollectorRegistry = CollectorRegistry()
_G_HEALTH = Gauge("bot_health", "Basic liveness indicator", registry=_METRICS)
_G_WIN7 = Gauge("win_rate_7d", "7d average win-rate (0..1)", registry=_METRICS)
_G_DD = Gauge("drawdown_pct", "Latest drawdown percentage", registry=_METRICS)

# init liveness
_G_HEALTH.set(1.0)


@app.get("/metrics")
def metrics():
    """
    Prometheus-eksponering. Bruges af testen tests/test_metrics_exposition.py
    og kan scrapes af Prometheus i drift.
    """
    data = generate_latest(_METRICS)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------
# Helpers
# ---------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV-læser:
    - on_bad_lines='skip' (skipper skæve linjer)
    - engine='python' (mere tolerant parser)
    - fallback: filtrér linjer der i det mindste ligner CSV (indeholder komma)
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        # ultra-robust fallback
        try:
            from io import StringIO

            with path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for ln in f:
                    # behold header + alle linjer der har mindst ét komma
                    if ("," in ln) or ln.lower().startswith("date") or ln.lower().startswith("ts"):
                        lines.append(ln)
            if not lines:
                return pd.DataFrame()
            return pd.read_csv(StringIO("".join(lines)), engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _safe_float(x, default: float = 0.0) -> float:
    """Konverter til float og gør NaN/Inf JSON-venlige."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _sanitize_json(obj):
    """
    Recursiv JSON-sanitizer til dict/list/primitive:
    - pandas.Timestamp/datetime/date -> ISO8601 streng (UTC-naiv)
    - numpy.number/bool/ndarray -> native Python
    - NaN/Inf -> None
    """
    # dict / list
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]

    # datotyper
    if isinstance(obj, pd.Timestamp):
        dt = obj.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.isoformat()
    if isinstance(obj, datetime):
        dt = obj
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()

    # numpy → native
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas NA (inkl. None -> False i isna, derfor try/except)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def _df_json_records(df: pd.DataFrame) -> list[dict]:
    """
    Gør en DataFrame JSON-sikker:
    - Datetime-kolonner -> ISO8601 (UTC-naiv)
    - NaN/Inf -> None
    - numpy-typer -> native
    """
    if df is None or df.empty:
        return []

    df = df.copy()

    # Konverter datetime-kolonner
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            s = df[col]
            # hvis tz-aware → konverter til UTC og drop tz
            try:
                if getattr(s.dt, "tz", None) is not None:
                    s = s.dt.tz_convert("UTC")
                s = s.dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                s = pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
            df[col] = s

    # NaN/Inf -> None
    df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)

    records = df.to_dict(orient="records")
    # Ekstra sikkerhed for afledte typer
    return jsonable_encoder(_sanitize_json(records))


def _read_last_run_ts() -> str:
    """Læs sidste behandlet bar fra fil – fallback til nu (UTC)."""
    if LAST_TS_FILE.exists():
        with contextlib.suppress(Exception):
            s = LAST_TS_FILE.read_text(encoding="utf-8").strip()
            if s:
                # behold ISO-format som er skrevet af live.py
                return s
    return datetime.now(timezone.utc).isoformat()


# ---------------------------
# Core Endpoints
# ---------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "version": app.version}


@app.get("/equity", response_model=List[EquityPoint])
def get_equity():
    df = _read_csv(LOGS / "equity.csv")
    if df.empty or "date" not in df.columns or "equity" not in df.columns:
        return []
    df = df.copy()
    # Datokolonne → datetime for sikker ISO-output
    with contextlib.suppress(Exception):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    records = _df_json_records(df[["date", "equity"]].dropna(subset=["equity"]))
    return records


@app.get("/metrics/daily", response_model=List[DailyMetric])
def get_daily_metrics(limit: int = Query(30, ge=1, le=90)):
    df = _read_csv(LOGS / "daily_metrics.csv")
    if df.empty:
        return []
    df = df.tail(limit).copy()

    # Konverter numeriske kolonner robust
    num_cols = [
        "signal_count",
        "trades",
        "win_rate",
        "gross_pnl",
        "net_pnl",
        "max_dd",
        "sharpe_d",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = (
                pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
            if c in {"signal_count", "trades"}:
                df[c] = df[c].astype(int)

    # Datokolonne til datetime for sikker ISO-output
    if "date" in df.columns:
        with contextlib.suppress(Exception):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return _df_json_records(df)


@app.get("/signals/latest")
def get_latest_signal(limit: int = Query(1, ge=1, le=200)):
    """
    Returnerer seneste signal(er) fra api/sim_signals.json.
    - Ved limit=1: returnér et objekt (seneste).
    - Ved limit>1: returnér en liste af objekter (seneste først).
    Bemærk: GUI håndterer både objekt og liste.
    """
    p = API_DIR / "sim_signals.json"
    if not p.exists():
        return None if limit == 1 else []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None if limit == 1 else []

    if not isinstance(data, list) or not data:
        return None if limit == 1 else []

    slice_ = data[-limit:][::-1]  # nyeste først
    payload = _sanitize_json(slice_)
    if limit == 1:
        return JSONResponse(content=payload[0])
    return JSONResponse(content=payload)


@app.get("/fills")
def get_fills(limit: int = Query(20, ge=1, le=500)):
    """
    Returnerer seneste handler fra logs/fills.csv.
    Kolonner håndteres 'best effort'; typisk: ts, symbol, side, qty, price, pnl_realized, commission.
    """
    p = LOGS / "fills.csv"
    if not p.exists():
        return JSONResponse(content=[])

    try:
        df = _read_csv(p)
    except Exception as e:
        return JSONResponse(content={"error": f"Kan ikke læse fills.csv: {e}"}, status_code=500)

    # Normaliser tid => 'ts' som datetime (senere til ISO)
    if "ts" in df.columns:
        with contextlib.suppress(Exception):
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(None)
    elif "timestamp" in df.columns:
        with contextlib.suppress(Exception):
            s = pd.to_numeric(df["timestamp"], errors="coerce")
            unit = "ms" if s.dropna().max() and float(s.dropna().max()) > 1e11 else "s"
            df["ts"] = pd.to_datetime(s, errors="coerce", unit=unit, utc=True).dt.tz_convert(None)

    df = df.sort_values("ts", ascending=False, na_position="last").reset_index(drop=True)
    if limit:
        df = df.head(limit)

    # Cast numeriske felter robust (hvis de findes)
    for c in ["qty", "price", "pnl_realized", "commission"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return JSONResponse(content=_df_json_records(df))


@app.get("/status")
def get_status():
    df_e = _read_csv(LOGS / "equity.csv")
    df_m = _read_csv(LOGS / "daily_metrics.csv")

    # last_run_ts fra fil (skrevet af live.py), fallback: nu (UTC)
    last_run_ts = _read_last_run_ts()

    # drawdown (sidste punkt)
    drawdown = 0.0
    if not df_e.empty and "drawdown_pct" in df_e.columns:
        dd_series = pd.to_numeric(df_e["drawdown_pct"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if len(dd_series) > 0:
            drawdown = _safe_float(dd_series.iloc[-1], 0.0)

    # win-rate (7d gennemsnit)
    win7 = 0.0
    if not df_m.empty and "win_rate" in df_m.columns:
        wr = pd.to_numeric(df_m["win_rate"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        tail = wr.tail(7)
        if not tail.isna().all():
            win7 = _safe_float(tail.mean(), 0.0)

    mode = "live" if os.getenv("LIVE_MODE", "paper").lower() == "live" else "paper"
    commit_sha = os.getenv("GIT_COMMIT") or None

    # ⬇️ NYT: opdatér Prometheus-gauges
    with contextlib.suppress(Exception):
        _G_HEALTH.set(1.0)
        _G_WIN7.set(float(win7))
        _G_DD.set(float(drawdown))

    payload = {
        "last_run_ts": last_run_ts,
        "mode": mode,
        "win_rate_7d": round(win7, 2),
        "drawdown_pct": round(drawdown, 2),
        "version": app.version,
        "commit_sha": commit_sha,
    }
    return JSONResponse(content=_sanitize_json(payload))


# ---------------------------
# AI / Claude integration
# ---------------------------
def _ask_claude(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    """
    Minimal wrapper med fail-fast:
    - AI_FORCE_FALLBACK=1 => bypass netværk, returnér offline-tekst
    - AI_HTTP_TIMEOUT (sek, float) styrer HTTP-timeout (default 8.0)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    force_fallback = os.getenv("AI_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
    if not api_key or force_fallback:
        return (
            "(AI slået fra) Sæt ANTHROPIC_API_KEY i miljøet for at aktivere forklaringer.\n"
            "Jeg kan stadig vise dine data, men ikke generere AI-forklaringer."
        )

    http_client = None
    try:
        # pip install anthropic
        from anthropic import Anthropic

        # Forsøg at styre timeout via httpx
        client: Optional[Anthropic] = None
        try:
            import httpx  # type: ignore

            http_timeout = float(os.getenv("AI_HTTP_TIMEOUT", "8.0"))
            http_client = httpx.Client(timeout=httpx.Timeout(http_timeout))
            client = Anthropic(api_key=api_key, http_client=http_client, max_retries=0)
        except Exception:
            # Fallback: standard klient (Anthropic har egen httpx internt)
            client = Anthropic(api_key=api_key)

        model = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-latest")
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = []
        for block in getattr(resp, "content", []) or []:
            # block.type == "text"
            txt = getattr(block, "text", None) or getattr(block, "value", None)
            if txt:
                parts.append(str(txt))
        text = "\n".join(parts).strip()
        return text or "(Tomt AI-svar)"
    except Exception as e:
        return f"(AI timeout/fejl) {e}\n(Viser offline-data i stedet.)"
    finally:
        try:
            if http_client is not None:
                http_client.close()
        except Exception:
            pass


@app.get("/ai/explain_trade")
def ai_explain_trade(i: int = 0, context_bars: int = 60):
    """
    Forklar handel nr. i (0 = seneste) baseret på logs:
    - finder raden i fills.csv
    - sampler seneste signaler & equity til kontekst
    - spørger Claude om kort forklaring + 2 risikonoter
    """
    fills_p = LOGS / "fills.csv"
    if not fills_p.exists():
        return {"text": "Ingen fills.csv fundet – kan ikke forklare en handel."}

    try:
        fills = _read_csv(fills_p)
    except Exception as e:
        return {"text": f"Kan ikke læse fills.csv: {e}"}

    if "ts" in fills.columns:
        with contextlib.suppress(Exception):
            fills["ts"] = pd.to_datetime(fills["ts"], errors="coerce", utc=True).dt.tz_convert(None)
    fills = fills.sort_values("ts", ascending=False, na_position="last").reset_index(drop=True)
    if len(fills) == 0:
        return {"text": "Ingen handler at forklare."}

    idx = max(0, min(int(i), len(fills) - 1))
    row = fills.iloc[idx].to_dict()

    # Kontekst – seneste signaler (fra JSON)
    signals = []
    sim_p = API_DIR / "sim_signals.json"
    if sim_p.exists():
        with contextlib.suppress(Exception):
            arr = json.loads(sim_p.read_text(encoding="utf-8"))
            if isinstance(arr, list):
                signals = arr[-min(len(arr), 50) :]

    # Kontekst – equity (seneste context_bars rækker)
    equity = []
    with contextlib.suppress(Exception):
        eq = _read_csv(LOGS / "equity.csv")
        equity = _df_json_records(eq.tail(max(1, int(context_bars))))

    payload = {
        "selected_trade": row,
        "recent_signals": signals,
        "equity_last": equity,
    }
    safe_payload = _sanitize_json(payload)

    system = "Du er en nøgtern trading-assistent. " "Svar korte bullets, ingen hype, ingen pynt."
    user = (
        "Forklar den udvalgte handel i højst 6 bullets (hvad, hvorfor, timing). "
        "Afslut med 2 konkrete risikonoter. Data (JSON):\n\n"
        + json.dumps(safe_payload, ensure_ascii=False)[:12000]
    )
    text = _ask_claude(system, user)
    return {"text": text}


@app.get("/ai/summary")
def ai_summary(limit_days: int = Query(7, ge=1, le=30)):
    """
    Lille statusopsummering (seneste 'limit_days' metrikrækker) + sidste signal.
    Brugbar til en hurtig 'daglig note'.
    """
    df_m = _read_csv(LOGS / "daily_metrics.csv")
    metrics = []
    if not df_m.empty:
        metrics = _df_json_records(df_m.tail(limit_days))

    # sidste signal (fra JSON)
    last_sig = None
    sim_p = API_DIR / "sim_signals.json"
    if sim_p.exists():
        with contextlib.suppress(Exception):
            arr = json.loads(sim_p.read_text(encoding="utf-8"))
            if isinstance(arr, list) and arr:
                last_sig = arr[-1]

    payload = {
        "metrics_tail": metrics,
        "last_signal": last_sig,
    }
    safe_payload = _sanitize_json(payload)

    system = "Du er en kortfattet assistent for en trader. " "Opsummer nøgletal og momentum kort."
    user = (
        "Lav en kort status (3–5 bullets) for trading-systemet, baseret på data (JSON):\n\n"
        + json.dumps(safe_payload, ensure_ascii=False)[:12000]
    )
    text = _ask_claude(system, user)
    return {"text": text}
