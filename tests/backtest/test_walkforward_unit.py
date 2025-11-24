# tests/backtest/test_walkforward_unit.py

import numpy as np
import pandas as pd

from backtest.walkforward import (
    WalkforwardConfig,
    assign_volatility_regimes,
    run_walkforward_with_regimes,
)


def _make_synthetic_df(n: int = 300) -> pd.DataFrame:
    """
    Simpelt syntetisk OHLCV-datasæt med varierende volatilitet, så vi kan få flere regimer.
    """
    np.random.seed(42)
    # Skiftende vol: lav, høj, lav, høj, ...
    vols = np.tile([0.002, 0.01], n // 2 + 1)[:n]
    noise = np.random.randn(n) * vols
    price = 100 + np.cumsum(noise)
    ts = pd.date_range(start="2020-01-01", periods=n, freq="H")

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": np.ones(n),
        }
    )
    return df


def test_assign_volatility_regimes_minimum_5_regimes():
    df = _make_synthetic_df(400)
    regimes = assign_volatility_regimes(df, n_regimes=5, close_col="close")

    assert len(regimes) == len(df)
    assert regimes.isna().sum() == 0
    # Med vores konstruerede vol-struktur bør vi få mindst 5 regimer
    assert regimes.nunique() >= 5


def test_run_walkforward_with_regimes_writes_reports(tmp_path, monkeypatch):
    """
    End-to-end smoke-test for B2:
    - Patcher run_backtest så vi ikke er afhængige af den konkrete implementering.
    - Tjekker at der kommer en ikke-tom regime-rapport og at filer skrives.
    """
    from backtest import walkforward as wf_mod

    def fake_run_backtest(df_window, signals_window):
        n = len(df_window)
        # Simple balancer, der vokser lidt over tid
        balance_df = pd.DataFrame(
            {
                "timestamp": df_window["timestamp"].reset_index(drop=True),
                "balance": np.linspace(100.0, 110.0, n),
            }
        )
        trades_df = pd.DataFrame(
            {
                "timestamp": df_window["timestamp"].reset_index(drop=True),
                "profit": np.linspace(0.0, 0.01, n),
                "type": ["CLOSE"] * n,
            }
        )
        return trades_df, balance_df

    # Patch den run_backtest som walkforward-modulet bruger
    monkeypatch.setattr(wf_mod, "run_backtest", fake_run_backtest)

    df = _make_synthetic_df(240)
    signals = np.ones(len(df), dtype=int)

    cfg = WalkforwardConfig(train_bars=80, oos_bars=40, step_bars=40, min_oos_trades=0)

    result = run_walkforward_with_regimes(
        df,
        signals,
        cfg,
        symbol="TESTCOIN",
        timeframe="1h",
        out_dir=str(tmp_path),
    )

    regimes_df = result["regimes"]
    folds_df = result["folds"]

    # Der skal være mindst én fold og mindst ét regime i rapporten
    assert not folds_df.empty
    assert not regimes_df.empty
    assert regimes_df["regime"].nunique() >= 1

    # Filer skal være skrevet der hvor funktionen rapporterer
    from pathlib import Path

    assert Path(result["folds_path"]).exists()
    assert Path(result["regimes_path"]).exists()
    assert Path(result["meta_path"]).exists()
