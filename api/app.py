# api/app.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
API_DIR = ROOT / "api"

app = FastAPI(title="AI Trading Bot API (Paper Demo)", version="0.1.0")

# Tillad Streamlit lokalt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class Signal(BaseModel):
    timestamp: int
    symbol: str
    side: str
    confidence: float
    price: float
    sl: float
    tp: float
    regime: str

class Status(BaseModel):
    last_run_ts: str
    mode: str
    win_rate_7d: float
    drawdown_pct: float
    version: str
    commit_sha: Optional[str] = None

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@app.get("/equity", response_model=List[EquityPoint])
def get_equity():
    df = _read_csv(LOGS / "equity.csv")
    if df.empty:
        return []
    return [{"date": str(r["date"]), "equity": float(r["equity"])} for _, r in df.iterrows()]

@app.get("/metrics/daily", response_model=List[DailyMetric])
def get_daily_metrics(limit: int = Query(30, ge=1, le=90)):
    df = _read_csv(LOGS / "daily_metrics.csv")
    if df.empty:
        return []
    df = df.tail(limit)
    records = df.to_dict(orient="records")
    return records  # FastAPI+pydantic caster

@app.get("/signals/latest", response_model=Signal | None)
def get_latest_signal():
    p = API_DIR / "sim_signals.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not data:
        return None
    return data[-1]

@app.get("/status", response_model=Status)
def get_status():
    df_e = _read_csv(LOGS / "equity.csv")
    df_m = _read_csv(LOGS / "daily_metrics.csv")
    last_run_ts = datetime.now(timezone.utc).isoformat()
    drawdown = float(df_e.iloc[-1]["drawdown_pct"]) if not df_e.empty else 0.0
    win7 = float(df_m.tail(7)["win_rate"].mean()) if not df_m.empty else 0.0
    return Status(
        last_run_ts=last_run_ts,
        mode="paper",
        win_rate_7d=round(win7, 2),
        drawdown_pct=round(drawdown, 2),
        version="0.1.0",
        commit_sha=None,
    )
