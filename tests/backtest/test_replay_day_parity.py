# tests/backtest/test_replay_day_parity.py

# -*- coding: utf-8 -*-
"""
Tests for B3 – Replay-dag paritet.

Vi bruger syntetiske OHLCV-data over flere dage, genererer simple signaler,
og verificerer at:
- dag-afkast fra fuld backtest == dag-afkast fra replay-dag (inden for tolerance)
- days uden trades giver 0-afkast i både full og replay
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.replay_day import check_replay_parity, replay_day_from_df


def _make_synthetic_ohlcv(n_days: int = 3, bars_per_day: int = 24) -> pd.DataFrame:
    """
    Generér et lille syntetisk OHLCV-sæt.

    - Start: 2021-01-01 00:00
    - Frekvens: 1H
    - Pris: en simpel trend + lidt variation for at sikre trades har effekt.
    """
    total = n_days * bars_per_day
    idx = pd.date_range("2021-01-01", periods=total, freq="H")

    # Enkel lineær trend + lidt “bølge”
    base = np.linspace(100.0, 110.0, total)
    wave = 0.5 * np.sin(np.linspace(0, 10, total))
    close = base + wave

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.ones(total),
        }
    )
    return df


def _make_signals_for_middle_day(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    Lav simple long-only signaler:
    - Vi åbner/holder long fra kl. 10 til 16 på “midter-dagen”.
    """
    timestamps = pd.to_datetime(df["timestamp"])
    dates = sorted({ts.date() for ts in timestamps})
    assert len(dates) >= 3, "Forventet mindst 3 dage i synthetic data."

    middle_day = dates[1]  # dag 2
    sig = np.zeros(len(df), dtype=int)

    for i, ts in enumerate(timestamps):
        if ts.date() == middle_day and 10 <= ts.hour <= 16:
            sig[i] = 1

    day_str = middle_day.strftime("%Y-%m-%d")
    return sig, day_str


def _make_all_flat_signals(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    Ingen trades på midter-dagen: alle signaler = 0.
    """
    timestamps = pd.to_datetime(df["timestamp"])
    dates = sorted({ts.date() for ts in timestamps})
    assert len(dates) >= 3, "Forventet mindst 3 dage i synthetic data."

    middle_day = dates[1]  # dag 2
    sig = np.zeros(len(df), dtype=int)
    day_str = middle_day.strftime("%Y-%m-%d")
    return sig, day_str


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_replay_parity_enkel_dag_matcher_full_backtest():
    """
    For en midter-dag med faktiske trades skal replay-dag give samme
    dagsafkast som dag-udsnittet fra en fuld backtest (inden for lille tolerance).
    """
    df = _make_synthetic_ohlcv(n_days=3, bars_per_day=24)
    signals, day = _make_signals_for_middle_day(df)

    res = check_replay_parity(
        df,
        signals,
        day=day,
        equity_tol_abs=1e-9,
        equity_tol_pct=1e-6,
    )

    # Kontrakt: ok-flagget skal være True, og afkastene må kun afvige numerisk minimalt.
    assert res["ok"] is True
    assert abs(res["ret_full_day"] - res["ret_replay"]) < 1e-9

    # Vi forventer faktiske trades på den dag
    assert res["n_trades_full_day"] >= 1
    assert res["n_trades_replay"] >= 1


def test_replay_parity_ingen_trades_giver_nul_afkast():
    """
    Hvis der ikke er nogen trades på dagen, skal både full backtest og replay
    rapportere ~0 dagsafkast, og paritet skal stadig være OK.
    """
    df = _make_synthetic_ohlcv(n_days=3, bars_per_day=24)
    signals, day = _make_all_flat_signals(df)

    # Kør først “rå” replay, så vi også indirekte tester replay_day_from_df
    trades_rep, balance_rep = replay_day_from_df(df, signals, day)
    assert trades_rep.empty or len(trades_rep) == 0

    res = check_replay_parity(
        df,
        signals,
        day=day,
        equity_tol_abs=1e-9,
        equity_tol_pct=1e-6,
    )

    # Ingen trades → 0-afkast i begge veje og paritet OK
    assert res["ok"] is True
    assert abs(res["ret_full_day"]) < 1e-9
    assert abs(res["ret_replay"]) < 1e-9
    assert res["n_trades_full_day"] == 0
    assert res["n_trades_replay"] == 0
