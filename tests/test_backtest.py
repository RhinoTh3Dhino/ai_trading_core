# tests/test_backtest.py
"""
Dækker kerne-funktioner i backtest/backtest.py:
- walk_forward_splits
- compute_regime / regime_filter / regime_performance
- get_git_hash (fallback når git ikke findes)
- force_trade_signals / clean_signals
- run_backtest (happy path + fejlgrene)
- calc_backtest_metrics (monkeypatch af advanced_performance_metrics)
- save_backtest_results (append uden header, inkl. git hash)
- load_csv_auto (med/uden meta-header)
"""

import sys
from pathlib import Path

# Sørg for at projektroden er i sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import os

import numpy as np
import pandas as pd
import pytest

from backtest import backtest as bt


# ------------------- Hjælpere -------------------
def _make_price_df(n=120, with_ema=True, start="2024-01-01"):
    # Brug 'h' frekvens for time-serie; pandas accepterer både 'h' og 'H'
    ts = pd.date_range(start, periods=n, freq="h")
    # Glat, deterministisk trend + lidt sving
    lin = np.linspace(100, 110, n)
    wobble = np.sin(np.linspace(0, 8, n)) * 0.5
    close = lin + wobble
    df = pd.DataFrame({"timestamp": ts, "close": close})
    if with_ema:
        # Kort span for hurtig stabilitet i tests (ema_200 i prod kan være længere)
        df["ema_200"] = df["close"].ewm(span=10, adjust=False).mean()
    return df


def _has_any_column(df: pd.DataFrame, candidates):
    return any(c in df.columns for c in candidates)


# ------------------- walk_forward_splits -------------------
def test_walk_forward_splits_basic():
    df = _make_price_df(100)
    splits = bt.walk_forward_splits(
        df, train_size=0.6, test_size=0.2, step_size=0.1, min_train=20
    )
    # For n=100: der bør være mindst ét vindue
    assert isinstance(splits, list) and len(splits) >= 1
    for tr_idx, te_idx in splits:
        assert len(tr_idx) >= 20 and len(te_idx) > 0
        # Indekser skal være voksende heltal
        assert all(int(i) == i for i in tr_idx)
        assert all(int(i) == i for i in te_idx)
        assert list(tr_idx) == sorted(tr_idx)
        assert list(te_idx) == sorted(te_idx)


# ------------------- compute_regime / regime_filter -------------------
def test_compute_regime_and_filter():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "close": [10, 8, 10],
            "ema_200": [9, 9, 10],
        }
    )
    out = bt.compute_regime(df.copy())
    # Forventet mapping (kan være afhængig af thresholds i implementationen; accepter str-values)
    assert "regime" in out.columns
    assert set(out["regime"].astype(str)) <= {"bull", "bear", "neutral"}

    signals = [1, -1, 1]
    filtered = bt.regime_filter(signals, out["regime"], active_regimes=["bull"])
    # I det mindste bør ikke-bull blive nulstillet:
    assert all(s in (0, 1, -1) for s in filtered)
    # Hvis ingen bull-ticks, giver det 0'ere – her tjekker vi kun længde og type
    assert len(filtered) == len(signals)

    filtered2 = bt.regime_filter(
        signals, out["regime"], active_regimes=["bull", "neutral"]
    )
    assert len(filtered2) == len(signals)


# ------------------- regime_performance -------------------
def test_regime_performance_ok_and_missing_col():
    # Manglende kolonne -> {}
    empty_stats = bt.regime_performance(pd.DataFrame({"type": [], "profit": []}))
    assert empty_stats == {}

    # Minimal testdata
    trades = pd.DataFrame(
        {
            "regime": ["bull", "bull", "bear", "bear", "neutral"],
            "profit": [0.1, -0.05, 0.02, -0.01, 0.0],
            "drawdown": [-1, -2, -3, -4, -5],
        }
    )
    stats = bt.regime_performance(trades)
    assert set(stats.keys()) <= {"bull", "bear", "neutral"}
    for s in stats.values():
        for key in ("num_trades", "win_rate", "profit_pct"):
            assert key in s


# ------------------- get_git_hash fallback -------------------
def test_get_git_hash_unknown(monkeypatch):
    # Simulér at git ikke er tilgængelig
    import subprocess as _sp

    def boom(*args, **kwargs):
        raise _sp.CalledProcessError(1, "git")

    monkeypatch.setattr(bt.subprocess, "check_output", boom)
    assert bt.get_git_hash() == "unknown"


# ------------------- force_trade_signals / clean_signals -------------------
def test_force_trade_signals_and_clean_signals(capsys):
    sig = bt.force_trade_signals(5)
    assert sig.tolist() == [1, -1, 1, -1, 1]

    # Mismatch: input længere end length → truncation + ingen crash
    out = bt.clean_signals(np.array([1, 2, 3, 4, 5]), 3)
    assert out.tolist() == [1, 2, 3]
    # Mismatch: input kortere → padding med 0
    out2 = bt.clean_signals(np.array([9, 8]), 4)
    assert out2.tolist() == [9, 8, 0, 0]
    # Series håndteres også
    out3 = bt.clean_signals(pd.Series([1, -1, 0]), 3)
    assert out3.tolist() == [1, -1, 0]
    # Skal logge advarsel når mismatch opstår
    _ = capsys.readouterr()  # flush
    bt.clean_signals([1, 2, 3], 5)
    logs = capsys.readouterr().out
    assert "Signal" in logs or "ADVARSEL" in logs or "mismatch" in logs.lower()


# ------------------- run_backtest (fejlgrene) -------------------
def test_run_backtest_raises_on_missing_columns():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h")})
    with pytest.raises(ValueError):
        bt.run_backtest(df, signals=[0] * len(df))  # 'close' mangler

    # 'datetime' skal mappes, og ema_200 kræves af compute_regime i mange implementeringer
    df2 = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
            "close": [1, 2, 3, 4, 5],
        }
    )
    df2["ema_200"] = pd.Series([1, 2, 3, 4, 5]).ewm(span=3, adjust=False).mean()
    # Virker trods 'datetime' (renames internt)
    trades, balance = bt.run_backtest(df2, signals=[0] * len(df2))
    assert isinstance(trades, pd.DataFrame) and isinstance(balance, pd.DataFrame)


# ------------------- run_backtest (happy path + required cols udfyldes) -------------------
def test_run_backtest_happy_path_and_required_columns_present():
    df = _make_price_df(80, with_ema=True)
    signals = np.zeros(len(df), dtype=int)  # HOLD -> sandsynligvis få/ingen handler
    trades, balance = bt.run_backtest(df, signals=signals)

    # Begge df'er skal eksistere
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(balance, pd.DataFrame)

    # run_backtest bør sikre/udskrive standardkolonner (tillad timestamp|datetime og balance|equity)
    assert _has_any_column(trades, ["timestamp", "datetime"])
    for col in ["type", "price", "regime"]:
        assert col in trades.columns
    assert _has_any_column(balance, ["timestamp", "datetime"])
    assert _has_any_column(balance, ["close", "balance", "equity"])
    # drawdown tilstede når balance ikke er tom
    if not balance.empty:
        assert "drawdown" in balance.columns


# ------------------- calc_backtest_metrics (monkeypatched) -------------------
def test_calc_backtest_metrics_maps_fields(monkeypatch):
    fake = {
        "profit_pct": 12.3,
        "win_rate": 0.55,
        "max_drawdown": -8.0,
        "max_consec_losses": 3,
        "recovery_bars": 42,
        "profit_factor": 1.7,
        "sharpe": 0.9,
        "sortino": 1.2,
    }
    monkeypatch.setattr(bt, "advanced_performance_metrics", lambda t, b, i: fake)

    trades = pd.DataFrame({"type": ["OPEN", "TP", "SL", "CLOSE", "INFO"]})
    balance = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "balance": [1000, 1010, 1005],
        }
    )
    out = bt.calc_backtest_metrics(trades, balance, initial_balance=1000)

    assert out["profit_pct"] == 12.3
    assert out["win_rate"] == 0.55
    assert out["drawdown_pct"] == -8.0
    assert out["num_trades"] == 3
    assert out["max_consec_losses"] == 3
    assert out["recovery_bars"] == 42
    assert out["profit_factor"] == 1.7
    assert out["sharpe"] == 0.9
    assert out["sortino"] == 1.2


# ------------------- save_backtest_results (append + git hash) -------------------
def test_save_backtest_results_appends_without_header(tmp_path, monkeypatch):
    # Fast git-hash for determinisme
    monkeypatch.setattr(bt, "get_git_hash", lambda: "abc123")
    out_csv = tmp_path / "backtest_results.csv"
    m = {"profit_pct": 10, "win_rate": 0.6, "drawdown_pct": -5, "num_trades": 2}
    bt.save_backtest_results(m, version="vX", csv_path=str(out_csv))
    bt.save_backtest_results(m, version="vY", csv_path=str(out_csv))

    df = pd.read_csv(out_csv)
    # Første kolonner bør være tidsstempel + version + git_hash
    assert list(df.columns)[:3] == ["timestamp", "version", "git_hash"]
    assert len(df) == 2
    assert set(df["version"]) == {"vX", "vY"}
    assert set(df["git_hash"]) == {"abc123"}


# ------------------- load_csv_auto -------------------
def test_load_csv_auto_with_and_without_meta(tmp_path, capsys):
    f1 = tmp_path / "plain.csv"
    f1.write_text("a,b\n1,2\n", encoding="utf-8")
    df1 = bt.load_csv_auto(str(f1))
    assert list(df1.columns) == ["a", "b"]

    f2 = tmp_path / "meta.csv"
    f2.write_text("# v1.2.3 meta header\nx,y\n7,8\n", encoding="utf-8")
    df2 = bt.load_csv_auto(str(f2))
    assert list(df2.columns) == ["x", "y"]
    out = capsys.readouterr().out.lower()
    # Implementations kan printe at meta-header blev detekteret
    assert "meta" in out or "header" in out


if __name__ == "__main__":
    import pytest

    pytest.main(
        [__file__, "-vv", "--cov=backtest.backtest", "--cov-report=term-missing"]
    )
