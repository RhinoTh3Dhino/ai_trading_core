# tests/test_backtest_metrics_more.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest import metrics as bm


def _make_base_df(n=10, start="2024-01-01 00:00:00"):
    ts = pd.date_range(start, periods=n, freq="H")
    # enkel regime-serie; 'bear' hver 3. time
    regime = np.where(np.arange(n) % 3 == 0, "bear", "bull")
    return pd.DataFrame({"timestamp": ts, "close": np.linspace(100, 101, n), "regime": regime})


def test_evaluate_strategies_uses_provided_trades_balance_and_merges_regime():
    df = _make_base_df(10)

    # Trades: to indenfor datasættet (matcher), en langt væk (ingen match -> 'ukendt')
    trades_df = pd.DataFrame(
        {
            "timestamp": [
                df["timestamp"].iloc[2],  # ~02:00 → match
                df["timestamp"].iloc[7],  # ~07:00 → match
                df["timestamp"].iloc[-1]
                + pd.Timedelta("10H"),  # udenfor tolerance → NaN → 'ukendt'
            ],
            "type": ["BUY", "TP", "SL"],
            "balance": [1000.0, 1010.0, 1000.0],
            # Tving 'regime_feat'-grenen ved at have en 'regime' i trades_df også
            "regime": ["dummy", "dummy", "dummy"],
        }
    )
    balance_df = pd.DataFrame(
        {
            "timestamp": trades_df["timestamp"],
            "balance": [1000.0, 1010.0, 1000.0],
        }
    )

    zeros = np.zeros(len(df), dtype=int)
    res = bm.evaluate_strategies(
        df,
        ml_signals=zeros,
        rsi_signals=zeros,
        macd_signals=zeros,
        ensemble_signals=zeros,
        trades_df=trades_df,
        balance_df=balance_df,
    )

    ens = res["ENSEMBLE"]
    assert ens["num_trades"] == len(trades_df)
    # Regime-statistik bør eksistere, inkl. 'ukendt' for den handel uden match
    stats = ens.get("regime_stats", {})
    assert isinstance(stats, dict) and stats != {}
    assert "ukendt" in stats  # dækker fillna-grenen


def test_evaluate_strategies_skips_regime_when_missing_on_df():
    df = _make_base_df(10).drop(columns=["regime"])  # fjern regime → do_regime bliver False

    trades_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
            ],
            "type": ["BUY", "TP"],
            "balance": [1000.0, 1010.0],
        }
    )
    balance_df = pd.DataFrame(
        {
            "timestamp": trades_df["timestamp"],
            "balance": [1000.0, 1010.0],
        }
    )

    zeros = np.zeros(len(df), dtype=int)
    res = bm.evaluate_strategies(
        df,
        ml_signals=zeros,
        rsi_signals=zeros,
        macd_signals=zeros,
        ensemble_signals=zeros,
        trades_df=trades_df,
        balance_df=balance_df,
    )
    assert res["ENSEMBLE"].get("regime_stats", {}) == {}  # springer analyse over


def test_evaluate_strategies_falls_back_to_run_and_score_when_no_trades_df():
    df = _make_base_df(12)
    zeros = np.zeros(len(df), dtype=int)

    # Ingen trades_df/balance_df → skal bruge run_and_score for ENSEMBLE
    res = bm.evaluate_strategies(
        df,
        ml_signals=zeros,
        rsi_signals=zeros,
        macd_signals=zeros,
        ensemble_signals=zeros,
        trades_df=None,
        balance_df=None,
    )

    ens = res["ENSEMBLE"]
    # run_and_score tilføjer altid 'num_trades' – typisk 0 med zeros-signaler
    assert "num_trades" in ens
    # fordi trades_df=None, skal evaluate_strategies også have sat regime_stats = {}
    assert ens.get("regime_stats", {}) == {}
