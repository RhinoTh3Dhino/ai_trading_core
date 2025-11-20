# tests/test_backtest_metrics_more_exceptions.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import backtest.metrics as bm


def test_evaluate_strategies_handles_merge_exception(monkeypatch):
    # Base DF med regime
    n = 6
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="H"),
            "close": np.linspace(100, 101, n),
            "regime": ["bull", "bear", "bull", "bear", "bull", "bear"],
        }
    )
    trades_df = pd.DataFrame(
        {
            "timestamp": [df["timestamp"].iloc[1], df["timestamp"].iloc[3]],
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

    # Tving merge_asof til at fejle → ram 'except' i evaluate_strategies
    monkeypatch.setattr(
        pd, "merge_asof", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    zeros = np.zeros(len(df), dtype=int)
    res = bm.evaluate_strategies(
        df, zeros, zeros, zeros, zeros, trades_df=trades_df, balance_df=balance_df
    )
    assert res["ENSEMBLE"].get("regime_stats", {}) == {}
