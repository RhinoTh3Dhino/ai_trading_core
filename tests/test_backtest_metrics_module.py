# tests/test_backtest_metrics_module.py
"""
Dækker backtest/metrics.py:
- calculate_sharpe
- calculate_drawdown
- run_and_score (monkeypatch af backtest.backtest.run_backtest / calc_backtest_metrics)
- regime_performance (mangler kolonne, numerisk mapping, tomme værdier)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest import metrics as m


# ------------------- calculate_sharpe -------------------
def test_calculate_sharpe_zero_variance_and_positive():
    assert m.calculate_sharpe([0, 0, 0]) == 0.0
    vals = [0.01, 0.02, 0.00, -0.01]
    out = m.calculate_sharpe(vals)
    assert isinstance(out, float)
    # ikke 0 når std != 0
    assert out != 0.0


# ------------------- calculate_drawdown -------------------
def test_calculate_drawdown_basic_and_empty():
    assert m.calculate_drawdown([]) == 0.0
    # 100 -> 110 -> 90 => max dd = (90-110)/110 = -0.1818...
    dd = m.calculate_drawdown([100, 110, 90])
    assert dd <= -0.18 and dd >= -0.20


# ------------------- run_and_score -------------------
def test_run_and_score_uses_backtest_and_counts_trades(monkeypatch):
    # monkeypatch run_backtest + calc_backtest_metrics inde i metrics-modulet
    def fake_run(df, signals):
        trades = pd.DataFrame({"type": ["TP", "SL", "OPEN"], "balance": [1000, 1010, 1005]})
        balance = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                "balance": [1000, 1010, 1005],
            }
        )
        return trades, balance

    def fake_calc(trades, balance):
        return {"profit_pct": 12.0, "win_rate": 0.5, "max_drawdown": -5.0}

    monkeypatch.setattr(m, "run_backtest", fake_run)
    monkeypatch.setattr(m, "calc_backtest_metrics", fake_calc)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "close": [1, 2, 3, 4, 5],
        }
    )
    sig = [0, 1, 0, -1, 0]
    out = m.run_and_score(df, sig)
    assert out["profit_pct"] == 12.0
    assert out["win_rate"] == 0.5
    assert out["num_trades"] == 3  # length af trades_df i fake_run


# ------------------- regime_performance -------------------
def test_regime_performance_missing_and_empty(capsys):
    # Manglende kolonne
    stats = m.regime_performance(pd.DataFrame({"type": []}))
    assert stats == {}
    msg = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "regime" in msg or "kolonne" in msg or "ikke fundet" in msg

    # Tilstede men tomme værdier
    stats2 = m.regime_performance(pd.DataFrame({"regime": [None, None]}))
    assert stats2 == {}
    msg2 = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "ingen regime" in msg2 or "ingen" in msg2


def test_regime_performance_numeric_mapping_and_values():
    # Numeriske regime-værdier → mappes til bull/bear/neutral
    trades = pd.DataFrame(
        {
            "regime": [0, 0, 1, 2, 2, 2],  # bull, bull, bear, neutral, neutral, neutral
            "type": ["TP", "SL", "TP", "SL", "TP", "TP"],
            "balance": [100, 101, 99, 100, 101, 103],
        }
    )
    stats = m.regime_performance(trades)
    # nøgler er str(...) af mappede værdier
    assert set(stats.keys()) == {"bull", "bear", "neutral"}
    # win_rate er TP/(TP+SL) pr. regime
    assert 0.0 <= stats["bull"]["win_rate"] <= 1.0
    assert 0.0 <= stats["bear"]["win_rate"] <= 1.0
    assert 0.0 <= stats["neutral"]["win_rate"] <= 1.0
    # profit_pct ≈ (last - first)/first * 100 når >=2 trades
    for k, v in stats.items():
        assert "num_trades" in v and "profit_pct" in v
