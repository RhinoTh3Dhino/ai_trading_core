# tests/test_backtest_walkforward_plot_real.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types

import numpy as np
import pandas as pd
import pytest

from backtest import backtest as bt


def _args(tmp_path):
    return types.SimpleNamespace(
        feature_path=str(tmp_path / "features.csv"),
        results_path=str(tmp_path / "backtest_results.csv"),
        balance_path=str(tmp_path / "balance.csv"),
        trades_path=str(tmp_path / "trades.csv"),
        strategy="ensemble",
        gridsearch=False,
        voting="majority",
        debug_ensemble=False,
        walkforward=True,
        train_size=0.6,
        test_size=0.2,
        step_size=0.2,
        force_trades=True,
    )


@pytest.mark.usefixtures()
def test_walkforward_real_plot(tmp_path, monkeypatch):
    # Skip pænt hvis matplotlib ikke er installeret i miljøet (robusthed)
    pytest.importorskip("matplotlib")

    # Syntetisk data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="H"),
            "close": np.linspace(100, 110, 50),
            "ema_200": np.linspace(99, 109, 50),
            "open": np.linspace(100, 110, 50),
            "high": np.linspace(101, 111, 50),
            "low": np.linspace(99, 109, 50),
            "volume": np.ones(50),
        }
    )

    # Patch args + loader
    monkeypatch.setattr(bt, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(bt, "load_csv_auto", lambda p: df.copy())

    # No-op for I/O/telegram, men lad pandas.plot køre rigtigt
    monkeypatch.setattr(bt, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_live_metrics", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_image", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_backtest_results", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_with_metadata", lambda *a, **k: None)

    # Kør (skal ramme hele plot-blokken og savefig)
    bt.main()
