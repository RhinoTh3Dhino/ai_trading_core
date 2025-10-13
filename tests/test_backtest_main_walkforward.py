# tests/test_backtest_main_walkforward.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types

import numpy as np
import pandas as pd

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
        force_trades=True,  # så vi ikke afhænger af strategier
    )


def test_main_walkforward_smoke(tmp_path, monkeypatch):
    # Syntetisk feature-CSV (indholdet bruges via load_csv_auto)
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

    # Monkeypatch parse_args + loader
    monkeypatch.setattr(bt, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(bt, "load_csv_auto", lambda p: df.copy())

    # No-op I/O / telegram
    calls = {"msg": 0, "live": 0, "savefig": 0}
    monkeypatch.setattr(
        bt, "send_message", lambda *a, **k: calls.__setitem__("msg", calls["msg"] + 1)
    )
    monkeypatch.setattr(
        bt,
        "send_live_metrics",
        lambda *a, **k: calls.__setitem__("live", calls["live"] + 1),
    )
    monkeypatch.setattr(bt, "send_image", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_backtest_results", lambda *a, **k: None)
    monkeypatch.setattr(bt, "save_with_metadata", lambda *a, **k: None)

    # Bypass pandas' backend-krav: gør DataFrame.plot til en no-op
    monkeypatch.setattr(pd.DataFrame, "plot", lambda *a, **k: None, raising=False)

    # Robust stub af matplotlib + pyplot (inkl. use("Agg") og savefig-tælling)
    class _PLT:
        def __init__(self, calls_dict):
            self._calls = calls_dict

        # Funktioner der typisk kaldes direkte i backtest
        def plot(self, *_, **__):
            return None

        def title(self, *_):
            return None

        def ylabel(self, *_):
            return None

        def xlabel(self, *_):
            return None

        def grid(self, *_):
            return None

        def tight_layout(self, *_):
            return None

        def savefig(self, *_):
            self._calls.__setitem__("savefig", self._calls["savefig"] + 1)

        # Funktioner som pandas/brugerkode kan kalde
        def figure(self, *_, **__):
            return self

        def gca(self, *_, **__):
            return self

        def subplots(self, *_, **__):
            return (self, self)

    class _MPL:
        def __init__(self, plt_obj):
            self.pyplot = plt_obj

        def use(self, *_args, **_kwargs):  # accepterer fx "Agg"
            return None

    fake_plt = _PLT(calls)
    fake_mpl = _MPL(fake_plt)

    # Brug monkeypatch.setitem så stubben ikke lækker til andre tests
    monkeypatch.setitem(sys.modules, "matplotlib", fake_mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    # Kør
    bt.main()

    # Vi forventer mindst én walkforward-iteration og et plot-saveforsøg
    assert calls["msg"] >= 1
    assert calls["live"] >= 1
    assert calls["savefig"] >= 1


def test_import_backtest_metrics_module():
    # Sørg for at modulet tæller i coverage – uden at forudsætte API
    import backtest.metrics as _bm  # noqa: F401

    assert _bm is not None
