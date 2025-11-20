# tests/test_backtest_extra_coverage.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types

import numpy as np
import pandas as pd

from backtest import backtest as bt


def _df(n=30, start="2024-01-01", freq="H", trend=0.1):
    idx = pd.date_range(start, periods=n, freq=freq)
    base = 100 + np.arange(n) * trend
    return pd.DataFrame({"timestamp": idx, "close": base})


def _args_grid(tmp_path):
    # Kører gridsearch-grenen og returnerer tidligt
    return types.SimpleNamespace(
        feature_path=str(tmp_path / "features.csv"),
        results_path=str(tmp_path / "backtest_results.csv"),
        balance_path=str(tmp_path / "balance.csv"),
        trades_path=str(tmp_path / "trades.csv"),
        strategy="ensemble",
        gridsearch=True,
        voting="majority",
        debug_ensemble=False,
        walkforward=False,
        train_size=0.6,
        test_size=0.2,
        step_size=0.2,
        force_trades=False,
    )


def _args_full(tmp_path):
    # Fuld-run uden walkforward/gridsearch
    return types.SimpleNamespace(
        feature_path=str(tmp_path / "features.csv"),
        results_path=str(tmp_path / "backtest_results.csv"),
        balance_path=str(tmp_path / "balance.csv"),
        trades_path=str(tmp_path / "trades.csv"),
        strategy="ensemble",
        gridsearch=False,
        voting="majority",
        debug_ensemble=False,
        walkforward=False,
        train_size=0.6,
        test_size=0.2,
        step_size=0.2,
        force_trades=False,
    )


def test_main_gridsearch_smoke_runs_and_calls_gridsearch(tmp_path, monkeypatch):
    # DF uden ema_200 -> rammer compute_regime fallback
    df = _df(24)

    # Monkeypatch CLI/IO
    monkeypatch.setattr(bt, "parse_args", lambda: _args_grid(tmp_path))
    monkeypatch.setattr(bt, "load_csv_auto", lambda p: df.copy())
    monkeypatch.setattr(bt, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_live_metrics", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_image", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_backtest_results", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_with_metadata", lambda *a, **k: None)

    # Tæl at gridsearch bliver kaldt
    calls = {"gs": 0}

    def _fake_grid(*args, **kwargs):
        calls["gs"] += 1
        # Tilstrækkeligt at returnere et ikke-tomt DataFrame (main() printer head og returnerer)
        return pd.DataFrame({"sl": [0.01], "tp": [0.02], "score": [1.0]})

    monkeypatch.setattr(bt, "grid_search_sl_tp_ema", _fake_grid)

    # Kør
    bt.main()

    assert calls["gs"] == 1  # gridsearch-grenen er ramt


def test_main_fullrun_force_debug_triggers_monitoring(tmp_path, monkeypatch):
    # DF uden ema_200 -> fallback i compute_regime
    df = _df(36)

    monkeypatch.setattr(bt, "parse_args", lambda: _args_full(tmp_path))
    monkeypatch.setattr(bt, "load_csv_auto", lambda p: df.copy())
    monkeypatch.setattr(bt, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(bt, "send_image", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_backtest_results", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_with_metadata", lambda *a, **k: None)

    calls = {"live": 0}
    monkeypatch.setattr(
        bt,
        "send_live_metrics",
        lambda *a, **k: calls.__setitem__("live", calls["live"] + 1),
    )

    # FORCE_DEBUG: undgår afhængighed af strategimoduler og sikrer handler
    monkeypatch.setattr(bt, "FORCE_DEBUG", True, raising=True)

    bt.main()

    assert calls["live"] >= 1  # fuld-run monitoring-grenen er ramt


def test_compute_regime_fallback_without_ema200():
    # Ingen ema_200-kolonne -> funktionen skal selv beregne EMA og producere 'regime'
    df = _df(10).drop(columns=[], errors="ignore")
    out = bt.compute_regime(df.copy())  # må ikke kaste, og skal tilføje 'regime'
    assert "regime" in out.columns
    # Fallbacken beregner også ema_200-kolonnen, så den bør eksistere uden NaN
    assert "ema_200" in out.columns
    assert out["regime"].notna().all()
    assert out["ema_200"].notna().all()
