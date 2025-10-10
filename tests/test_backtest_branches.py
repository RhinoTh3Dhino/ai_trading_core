# tests/test_backtest_branches.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib

import numpy as np
import pandas as pd

from backtest import backtest as bt


def _df(seq, start="2024-01-01 00:00:00"):
    ts = pd.date_range(start, periods=len(seq), freq="H")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": seq,
            "high": seq,
            "low": seq,
            "close": seq,
            "volume": np.ones(len(seq)),
            "ema_200": np.array(seq, dtype=float) - 0.1,  # undgå NaN-check
        }
    )


def test_long_tp_and_sl_and_close_on_last_bar():
    # Long TP (>= 1.2%)
    df_tp = _df([100.0, 101.3, 101.3])  # +1.3%
    trades_tp, bal_tp = bt.run_backtest(df_tp, signals=[1, 0, 0])
    assert any(trades_tp["type"].eq("TP")), "Forventede TP for long"
    assert len(bal_tp) == len(df_tp)

    # Long SL (<= -0.6%)
    df_sl = _df([100.0, 99.0, 99.0])  # -1.0%
    trades_sl, _ = bt.run_backtest(df_sl, signals=[1, 0, 0])
    assert any(trades_sl["type"].eq("SL")), "Forventede SL for long"

    # Close i sidste bar (entry på sidste bar)
    df_close = _df([100.0, 100.5])
    trades_close, _ = bt.run_backtest(df_close, signals=[0, 1])
    assert any(
        trades_close["type"].eq("CLOSE")
    ), "Åben position skal lukkes i sidste bar"


def test_short_tp_and_sl_paths():
    # Short TP: pris falder (positiv change for short)
    df_tp = _df([100.0, 98.5, 98.4])  # -1.5% ≈ TP
    trades_tp, _ = bt.run_backtest(df_tp, signals=[-1, 0, 0])
    assert any(trades_tp["type"].eq("TP")), "Forventede TP for short"

    # Short SL: pris stiger (negativ change for short)
    df_sl = _df([100.0, 101.0, 101.0])  # +1.0% ≈ SL
    trades_sl, _ = bt.run_backtest(df_sl, signals=[-1, 0, 0])
    assert any(trades_sl["type"].eq("SL")), "Forventede SL for short"


def test_force_dummy_trades_and_drawdown_interpolation(monkeypatch):
    # Ingen handler → tving dummy-trades
    df = _df([100.0, 100.0, 100.0, 100.0])
    orig = bt.FORCE_DUMMY_TRADES
    try:
        monkeypatch.setattr(bt, "FORCE_DUMMY_TRADES", True, raising=True)
        trades_df, balance_df = bt.run_backtest(df, signals=[0, 0, 0, 0])
        assert not trades_df.empty and {"BUY", "TP", "SL"}.issubset(
            set(trades_df["type"])
        )
        # drawdown interpoleres ind i trades_df
        assert "drawdown" in trades_df.columns
        assert trades_df["drawdown"].notna().any()
    finally:
        monkeypatch.setattr(bt, "FORCE_DUMMY_TRADES", orig, raising=True)
