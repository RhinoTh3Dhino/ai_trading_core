# tests/test_backtest_metrics_more_prints.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import backtest.metrics as bm


def test_evaluate_strategies_exception_prints(monkeypatch, capsys):
    # Lille df med regime
    n = 4
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="H"),
        "close": np.linspace(100, 101, n),
        "regime": ["bull", "bear", "bull", "bear"],
    })

    # Minimal trades/balance til at trigge regime-merge
    trades_df = pd.DataFrame({
        "timestamp": [df["timestamp"].iloc[1], df["timestamp"].iloc[2]],
        "type": ["BUY", "TP"],
        "balance": [1000.0, 1010.0],
    })
    balance_df = pd.DataFrame({
        "timestamp": trades_df["timestamp"],
        "balance": [1000.0, 1010.0],
    })

    # Tving merge_asof til at crashe → rammer except-grenen med print(trades_df.head()) og print(df[['timestamp','regime']].head())
    def boom(*_a, **_k):
        raise RuntimeError("boom")
    monkeypatch.setattr(pd, "merge_asof", boom, raising=True)

    zeros = np.zeros(len(df), dtype=int)
    bm.evaluate_strategies(df, zeros, zeros, zeros, zeros, trades_df=trades_df, balance_df=balance_df)

    out = (capsys.readouterr().out + capsys.readouterr().err)
    assert "Fejl under regime-analyse" in out
    # Verificér at begge prints kom med (kolonne-navnene er nok til at matche)
    assert "timestamp" in out
    assert "regime" in out
