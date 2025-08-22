# tests/test_backtest_strategies_switch.py
import sys, types
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from backtest import backtest as bt

def _args(tmp_path, strategy):
    return types.SimpleNamespace(
        feature_path=str(tmp_path / "features.csv"),
        results_path=str(tmp_path / "res.csv"),
        balance_path=str(tmp_path / "bal.csv"),
        trades_path=str(tmp_path / "trades.csv"),
        strategy=strategy, gridsearch=False, voting="majority",
        debug_ensemble=False, walkforward=False,
        train_size=0.6, test_size=0.2, step_size=0.2, force_trades=False
    )

def _df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="H"),
        "close": np.linspace(100, 101, 30),
        "ema_200": np.linspace(99, 100, 30),
        "open": np.linspace(100, 101, 30),
        "high": np.linspace(100.5, 101.5, 30),
        "low": np.linspace(99.5, 100.5, 30),
        "volume": np.ones(30),
    })

def _run_strategy(tmp_path, monkeypatch, strategy):
    # let loader
    monkeypatch.setattr(bt, "parse_args", lambda: _args(tmp_path, strategy))
    monkeypatch.setattr(bt, "load_csv_auto", lambda p: _df())

    # no-op output
    monkeypatch.setattr(bt, "save_backtest_results", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_with_metadata", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_live_metrics", lambda *a, **k: None)

    bt.main()

def test_main_strategy_voting(tmp_path, monkeypatch):
    _run_strategy(tmp_path, monkeypatch, "voting")

def test_main_strategy_regime(tmp_path, monkeypatch):
    _run_strategy(tmp_path, monkeypatch, "regime")

def test_main_strategy_ema_rsi(tmp_path, monkeypatch):
    _run_strategy(tmp_path, monkeypatch, "ema_rsi")

def test_main_strategy_meanrev(tmp_path, monkeypatch):
    _run_strategy(tmp_path, monkeypatch, "meanrev")
