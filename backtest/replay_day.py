# backtest/replay_day.py

# -*- coding: utf-8 -*-
"""
Replay-dag paritet: helper-modul til at teste, at en enkelt dags “replay”
giver samme resultater som dag-udsnittet fra en fuld backtest.

Kontrakt:
- Vi bruger eksisterende run_backtest(df, signals) som eneste backtest-motor.
- Vi normaliserer alle timestamps til datetime.
- Vi sammenligner afkast for én kalenderdag:
  - full-run: dag-afkast udledt fra balance/equity-kurven
  - replay: run_backtest på præcis samme dag (subset af df/signals)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from backtest.backtest import run_backtest  # type: ignore


@dataclass
class ReplayParityResult:
    day: str
    ret_full_day: float
    ret_replay: float
    diff_abs: float
    diff_pct: float
    n_trades_full_day: int
    n_trades_replay: int
    ok: bool

    def to_dict(self) -> Dict[str, float | int | bool | str]:
        return {
            "day": self.day,
            "ret_full_day": self.ret_full_day,
            "ret_replay": self.ret_replay,
            "diff_abs": self.diff_abs,
            "diff_pct": self.diff_pct,
            "n_trades_full_day": self.n_trades_full_day,
            "n_trades_replay": self.n_trades_replay,
            "ok": self.ok,
        }


# ---------------------------------------------------------------------------
# Intern helpers
# ---------------------------------------------------------------------------


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sørg for, at der findes en datetime-kolonne 'timestamp' og at data er sorteret.
    """
    out = df.copy()

    if "timestamp" not in out.columns and "datetime" in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})

    if "timestamp" not in out.columns:
        raise ValueError("Input DataFrame skal have kolonnen 'timestamp' eller 'datetime'.")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("Kunne ikke parse nogle timestamps i input DataFrame.")

    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _ensure_signals(signals, n: int) -> np.ndarray:
    """
    Konverter til numpy-array og tjek længde mod df.
    """
    sig = np.asarray(signals).astype(int)
    if sig.shape[0] != n:
        raise ValueError(f"Længde på signals ({sig.shape[0]}) matcher ikke df ({n}).")
    return sig


def _select_equity_series(balance_df: pd.DataFrame) -> pd.Series:
    """
    Vælg balance/equity-kolonne til afkastberegning.
    """
    if "balance" in balance_df.columns:
        return pd.to_numeric(balance_df["balance"], errors="coerce")
    if "equity" in balance_df.columns:
        return pd.to_numeric(balance_df["equity"], errors="coerce")
    raise ValueError("balance_df skal indeholde enten 'balance' eller 'equity'-kolonne.")


def _slice_balance_for_day(balance_df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Udsnit af balance_df for en bestemt kalenderdag (YYYY-MM-DD).
    """
    if "timestamp" not in balance_df.columns:
        raise ValueError("balance_df skal have 'timestamp'-kolonne til dag-filtrering.")

    b = balance_df.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
    if b["timestamp"].isna().any():
        raise ValueError("Kunne ikke parse nogle timestamps i balance_df.")

    target_date = pd.to_datetime(day).date()
    m = b["timestamp"].dt.date == target_date
    return b.loc[m].reset_index(drop=True)


def _daily_return_from_balance(balance_df: pd.DataFrame, day: str) -> Tuple[float, int]:
    """
    Beregn dagsafkast for en given dato baseret på balance/equity-kurven.

    Returnerer:
        (ret, n_points)
    hvor ret er simpel afkast-faktor-1 (fx 0.02 = +2%).
    """
    day_bal = _slice_balance_for_day(balance_df, day)
    if day_bal.empty:
        return 0.0, 0

    eq = _select_equity_series(day_bal).dropna()
    if eq.shape[0] < 2:
        return 0.0, int(eq.shape[0])

    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])
    if start <= 0:
        return 0.0, int(eq.shape[0])
    ret = (end / start) - 1.0
    return float(ret), int(eq.shape[0])


def _count_trades_for_day(trades_df: pd.DataFrame, day: str) -> int:
    """
    Tæl trades (fx CLOSE legs) for en bestemt dag, så vi kan se,
    om antallet matcher mellem full-run og replay.
    """
    if trades_df is None or trades_df.empty:
        return 0

    df = trades_df.copy()

    # timestamp-kolonne kan hedde 'timestamp' eller 'ts'
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        ts_col = "timestamp"
    elif "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce")
        ts_col = "timestamp"
    else:
        return 0

    day_date = pd.to_datetime(day).date()
    df = df[df[ts_col].dt.date == day_date]

    # Hvis der findes en 'type' kolonne, kan vi evt. fokusere på CLOSE legs
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.upper() == "CLOSE"]

    return int(len(df))


# ---------------------------------------------------------------------------
# Offentlige funktioner
# ---------------------------------------------------------------------------


def replay_day_from_df(
    df: pd.DataFrame,
    signals,
    day: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kør en “replay” backtest for én kalenderdag.

    Parametre
    ---------
    df : pd.DataFrame
        Fuldt datasæt med mindst kolonnerne:
        - timestamp/datetime
        - open, high, low, close, volume
    signals : array-like
        Signaler for hele df (0/1 eller -1/0/1), længde = len(df).
    day : str
        Dato i format 'YYYY-MM-DD'.

    Returnerer
    ----------
    trades_day : pd.DataFrame
        Trades for replay-dagen.
    balance_day : pd.DataFrame
        Balance/equity-kurve for replay-dagen.
    """
    df_norm = _ensure_timestamp(df)
    sig = _ensure_signals(signals, len(df_norm))

    day_date = pd.to_datetime(day).date()
    m = df_norm["timestamp"].dt.date == day_date

    df_day = df_norm.loc[m].reset_index(drop=True)
    sig_day = sig[m]

    if df_day.empty:
        # Ingen data for denne dag → tomme frames
        return (
            pd.DataFrame(columns=["timestamp", "price", "qty", "profit"]),
            pd.DataFrame(columns=["timestamp", "balance"]),
        )

    trades_day, balance_day = run_backtest(df_day, sig_day)
    return trades_day, balance_day


def check_replay_parity(
    df: pd.DataFrame,
    signals,
    day: str,
    equity_tol_abs: float = 1e-6,
    equity_tol_pct: float = 1e-3,
) -> Dict[str, float | int | bool | str]:
    """
    Sammenlign dag-afkast fra fuld backtest med dag-afkast fra replay-dag.

    Vi sammenligner **relative afkast** (ikke absolut equity), så det ikke
    betyder noget, at replay-kørslen starter fra en “ren” startkapital.

    Parametre
    ---------
    df : pd.DataFrame
        Fuldt datasæt.
    signals :
        Signaler for hele datasættet (længde = len(df)).
    day : str
        Dato 'YYYY-MM-DD' der skal sammenlignes.
    equity_tol_abs : float
        Absolut tolerance på afkast (fx 1e-6).
    equity_tol_pct : float
        Relativ tolerance på afkast, i fraktion (0.01 = 1%).

    Returnerer
    ----------
    dict
        Se ReplayParityResult.to_dict() for felter.
    """
    df_norm = _ensure_timestamp(df)
    sig = _ensure_signals(signals, len(df_norm))

    # Fuld backtest
    trades_full, balance_full = run_backtest(df_norm, sig)
    ret_full, _ = _daily_return_from_balance(balance_full, day)
    n_trades_full = _count_trades_for_day(trades_full, day)

    # Replay-dag
    trades_rep, balance_rep = replay_day_from_df(df_norm, sig, day)
    ret_rep, _ = _daily_return_from_balance(balance_rep, day)
    n_trades_rep = _count_trades_for_day(trades_rep, day)

    diff_abs = float(abs(ret_full - ret_rep))
    denom = max(abs(ret_full), 1e-12)
    diff_pct = float(diff_abs / denom)

    ok = (diff_abs <= equity_tol_abs) or (diff_pct <= equity_tol_pct)

    res = ReplayParityResult(
        day=day,
        ret_full_day=float(ret_full),
        ret_replay=float(ret_rep),
        diff_abs=diff_abs,
        diff_pct=diff_pct,
        n_trades_full_day=n_trades_full,
        n_trades_replay=n_trades_rep,
        ok=bool(ok),
    )
    return res.to_dict()
