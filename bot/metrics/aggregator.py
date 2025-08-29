# bot/metrics/aggregator.py
from __future__ import annotations

"""
Daglig metrics-aggregator for paper trading.

Kilder (alle valgfrie – aggregator er robust hvis noget mangler):
- logs/equity.csv  : kolonner ["date","equity","cash","positions_value","drawdown_pct"]
- logs/fills.csv   : kolonner ["ts","symbol","side","qty","price","commission","pnl_realized"]
- logs/signals.csv : kolonner ["ts","signal"]  (valgfri; bruges til "signal_count")

Output:
- logs/daily_metrics.csv med kolonner:
  ["date","signal_count","trades","win_rate","gross_pnl","net_pnl","max_dd","sharpe_d"]

Funktioner:
- aggregate(days=None, recompute_all=False, update_only=False) → opdaterer/indsætter rækker.
- load_daily_metrics(limit=30) → DataFrame til API/GUI.
- CLI:
    python -m bot.metrics.aggregator --days 30 --update-last
    python -m bot.metrics.aggregator --recompute-all
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- PROJECT_ROOT fallback ---
try:
    from utils.project_path import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGS_DIR = Path(PROJECT_ROOT) / "logs"
EQUITY_CSV = LOGS_DIR / "equity.csv"
FILLS_CSV = LOGS_DIR / "fills.csv"
SIGNALS_CSV = LOGS_DIR / "signals.csv"
DAILY_METRICS_CSV = LOGS_DIR / "daily_metrics.csv"

COLUMNS = ["date", "signal_count", "trades", "win_rate", "gross_pnl", "net_pnl", "max_dd", "sharpe_d"]


# -----------------------
# Hjælpere
# -----------------------
def _ensure_headers(path: Path, headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # korrupt? returnér tom
        return pd.DataFrame()


def _to_date_series(x: Iterable) -> pd.Series:
    s = pd.to_datetime(pd.Series(list(x)), errors="coerce", utc=True)
    return s.dt.tz_localize(None).dt.date


def _round(x: float, n: int = 2) -> float:
    q = 10**n
    return math.floor(float(x) * q + 0.5) / q


# -----------------------
# Kernelogik
# -----------------------
def _dates_from_equity() -> List[str]:
    df = _read_csv_safe(EQUITY_CSV)
    if df.empty:
        return []
    if "date" not in df.columns:
        return []
    d = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().unique().tolist()
    return [str(x) for x in sorted(d)]


def _dates_from_fills() -> List[str]:
    df = _read_csv_safe(FILLS_CSV)
    if df.empty or "ts" not in df.columns:
        return []
    d = _to_date_series(df["ts"]).dropna().unique().tolist()
    return [str(x) for x in sorted(d)]


def _dates_from_signals() -> List[str]:
    df = _read_csv_safe(SIGNALS_CSV)
    if df.empty or "ts" not in df.columns:
        return []
    d = _to_date_series(df["ts"]).dropna().unique().tolist()
    return [str(x) for x in sorted(d)]


def _signal_counts_per_day() -> Dict[str, int]:
    """
    Tæl 'entries' som antallet af 0→1 flip i 'signals.csv' pr. dato.
    Hvis filen ikke findes, returneres tom dict (GUI/API kan leve med 0).
    """
    df = _read_csv_safe(SIGNALS_CSV)
    if df.empty:
        return {}
    # understøt både 'timestamp' og 'ts'
    if "ts" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" not in df.columns or "signal" not in df.columns:
        return {}
    df = df.sort_values("ts").reset_index(drop=True)
    df["sig"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)
    df["date"] = _to_date_series(df["ts"])
    flips = (df["sig"].diff().fillna(df["sig"]) > 0).astype(int)  # 0->1 tælles
    out = df.groupby("date")[flips.name].sum().astype(int).to_dict()
    # Pandas navngiver serien "sig" → diff() mister navn; sikre nøgler i dict
    # derfor beregner vi igen simpelt:
    out = df.groupby("date")["sig"].apply(lambda s: int(((s.diff() > 0).fillna(s.iloc[0] > 0)).sum())).to_dict()
    # konverter nøgler til str
    return {str(k): int(v) for k, v in out.items()}


def _intraday_dd_and_sharpe_for_day(date_str: str) -> Tuple[float, float]:
    """
    Beregn intradag max drawdown (%) samt en enkel Sharpe_d ud fra equity-stødpunkter.
    """
    df = _read_csv_safe(EQUITY_CSV)
    if df.empty or "date" not in df.columns or "equity" not in df.columns:
        return 0.0, 0.0
    day = df[df["date"].astype(str) == date_str]
    if day.empty:
        return 0.0, 0.0

    e = pd.to_numeric(day["equity"], errors="coerce").dropna().values
    if e.size == 0:
        return 0.0, 0.0

    peak = -1e18
    dd_pct = 0.0
    for val in e:
        peak = max(peak, val)
        dd_pct = min(dd_pct, (val - peak) / (peak + 1e-12) * 100.0)

    rets = np.diff(e)
    sharpe_d = float(np.mean(rets) / np.std(rets)) if (rets.size > 1 and np.std(rets) > 1e-12) else 0.0
    return _round(dd_pct, 2), _round(sharpe_d, 2)


def _pnl_and_wins_for_day(date_str: str) -> Tuple[float, float, int, int]:
    """
    Returner: gross_pnl, commissions, wins, closed_trades_count for den pågældende dato.
    - 'closed_trades_count' tælles som antal fills med pnl_realized != 0 (lukkede mængder).
    """
    df = _read_csv_safe(FILLS_CSV)
    if df.empty or "ts" not in df.columns:
        return 0.0, 0.0, 0, 0
    day = df[_to_date_series(df["ts"]).astype(str) == date_str]
    if day.empty:
        return 0.0, 0.0, 0, 0

    pnl_real = pd.to_numeric(day.get("pnl_realized", 0.0), errors="coerce").fillna(0.0)
    commissions = pd.to_numeric(day.get("commission", 0.0), errors="coerce").fillna(0.0)

    gross = float(pnl_real.sum())
    wins = int((pnl_real > 0).sum())
    closed = int((pnl_real != 0).sum())
    comm = float(commissions.sum())
    return _round(gross, 2), _round(comm, 2), wins, closed


def _existing_signal_counts_from_metrics() -> Dict[str, int]:
    """
    Hvis daily_metrics.csv allerede indeholder 'signal_count', læs dem ind, så vi ikke
    overskriver manuelt satte værdier (fx fra engine).
    """
    df = _read_csv_safe(DAILY_METRICS_CSV)
    if df.empty or "date" not in df.columns or "signal_count" not in df.columns:
        return {}
    m = {}
    for _, r in df.iterrows():
        try:
            m[str(r["date"])] = int(r["signal_count"])
        except Exception:
            pass
    return m


def _union_dates(limit_days: Optional[int] = None) -> List[str]:
    all_dates = sorted(set(_dates_from_equity()) | set(_dates_from_fills()) | set(_dates_from_signals()))
    if limit_days is not None and limit_days > 0 and len(all_dates) > limit_days:
        # tag kun de sidste N
        all_dates = all_dates[-limit_days:]
    return all_dates


# -----------------------
# Public API
# -----------------------
def aggregate(
    days: Optional[int] = None,
    recompute_all: bool = False,
    update_only: bool = False,
) -> pd.DataFrame:
    """
    Aggreger daglige metrikker og skriv/merge til logs/daily_metrics.csv.

    Parametre:
        days          : Kun de seneste N kalenderdage (None = alle, afhænger af kilder).
        recompute_all : Hvis True, ignorerer eksisterende rækker og regner for alle datoer.
        update_only   : Hvis True, opdaterer kun eksisterende dato-rækker (nyttigt til EOD-job).

    Returnerer:
        DataFrame med (senest beregnede) rækker sorteret efter dato (stigende).
    """
    _ensure_headers(DAILY_METRICS_CSV, COLUMNS)

    existing_df = _read_csv_safe(DAILY_METRICS_CSV)
    existing_df = existing_df if not existing_df.empty else pd.DataFrame(columns=COLUMNS)

    # Find hvilke datoer vi skal processe
    if recompute_all:
        dates = _union_dates(limit_days=days)
    elif update_only and not existing_df.empty:
        # opdater kun de datoer, som allerede findes (evt. begrænset af 'days')
        dates = existing_df["date"].astype(str).tolist()
        if days is not None and days > 0:
            dates = dates[-days:]
    else:
        dates = _union_dates(limit_days=days)

    if not dates:
        return existing_df

    # signal counts – fra signals.csv eller eksisterende metrics (prioritet til signals.csv)
    signal_counts = _signal_counts_per_day()
    if not signal_counts:
        signal_counts = _existing_signal_counts_from_metrics()

    updated_rows: List[Dict[str, str | float | int]] = []
    for d in dates:
        gross, comm, wins, closed = _pnl_and_wins_for_day(d)
        net = _round(gross - comm, 2)
        win_rate = _round((wins / max(closed, 1)) * 100.0, 2)
        max_dd, sharpe_d = _intraday_dd_and_sharpe_for_day(d)
        sig_count = int(signal_counts.get(d, 0))

        updated_rows.append(
            {
                "date": d,
                "signal_count": sig_count,
                "trades": int(closed),
                "win_rate": win_rate,
                "gross_pnl": gross,
                "net_pnl": net,
                "max_dd": max_dd,
                "sharpe_d": sharpe_d,
            }
        )

    # Merge idempotent
    base = existing_df.set_index("date", drop=False)
    for row in updated_rows:
        base.loc[row["date"]] = row  # indsætter eller opdaterer
    base = base.reset_index(drop=True)

    # Sortér efter dato (stigende) og skriv
    base = base.sort_values("date").reset_index(drop=True)
    base[COLUMNS].to_csv(DAILY_METRICS_CSV, index=False)
    return base[COLUMNS]


def load_daily_metrics(limit: int = 30) -> pd.DataFrame:
    """Læs daily_metrics.csv og returnér de seneste N rækker (stigende dato)."""
    df = _read_csv_safe(DAILY_METRICS_CSV)
    if df.empty:
        _ensure_headers(DAILY_METRICS_CSV, COLUMNS)
        return pd.DataFrame(columns=COLUMNS)
    df = df.sort_values("date")
    if limit and limit > 0:
        df = df.tail(limit)
    # type-casts for sikkerhed
    for col in ["signal_count", "trades"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["win_rate", "gross_pnl", "net_pnl", "max_dd", "sharpe_d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float).round(2)
    return df[COLUMNS]


# -----------------------
# CLI
# -----------------------
def _print_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("Ingen data.")
        return
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df.to_string(index=False))


def main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Daglig metrics-aggregator (paper trading)")
    parser.add_argument("--days", type=int, default=None, help="Begræns til seneste N dage")
    parser.add_argument("--recompute-all", action="store_true", help="Genberegn alle datoer fra kilderne")
    parser.add_argument("--update-last", action="store_true", help="Opdater kun eksisterende dato-rækker (evt. begrænset af --days)")
    parser.add_argument("--print", action="store_true", help="Print resultatet efter skrivning")
    args = parser.parse_args()

    df = aggregate(days=args.days, recompute_all=args.recompute_all, update_only=args.update_last)
    if args.print:
        _print_df(df)
    else:
        print(f"✅ Skrev/opfreshede {len(df)} rækker til {DAILY_METRICS_CSV}")


if __name__ == "__main__":  # pragma: no cover
    main_cli()
