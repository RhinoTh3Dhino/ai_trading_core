import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backtest.backtest import run_backtest

def make_dummy_df(rows=10):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=rows, freq="h"),
        "close": np.linspace(100, 110, rows),
        "signal": np.random.choice([1, 0, -1], size=rows),
        "ema_9": np.linspace(100, 105, rows),
        "ema_21": np.linspace(100, 108, rows),
        "ema_50": np.linspace(100, 109, rows),
        "ema_200": np.linspace(100, 110, rows),
        "rsi_14": np.random.uniform(30, 70, rows),
        "macd": np.random.uniform(-2, 2, rows),
        "macd_signal": np.random.uniform(-2, 2, rows),
        "atr_14": np.random.uniform(0.5, 2, rows),
        "regime": np.random.choice(["bull", "bear"], size=rows)
    })
    # Sikrer mindst ét BUY og ét SELL
    df.loc[0, "signal"] = 1
    df.loc[1, "signal"] = -1
    return df

def test_run_backtest_returns_dataframes():
    df = make_dummy_df()
    trades_df, balance_df = run_backtest(df)
    assert isinstance(trades_df, pd.DataFrame), "trades_df er ikke en DataFrame"
    assert isinstance(balance_df, pd.DataFrame), "balance_df er ikke en DataFrame"
    assert not trades_df.empty, "trades_df er tom – testdata sikrer mindst én trade!"
    assert not balance_df.empty, "balance_df er tom – bør altid have balance-tracking!"
    # Centrale kolonner i trades_df (timestamp, type, price, balance, regime)
    for col in ["timestamp", "type", "price", "balance", "regime"]:
        assert col in trades_df.columns, f"{col} mangler i trades_df"
    # Centrale kolonner i balance_df (timestamp, close)
    for col in ["timestamp", "close"]:
        assert col in balance_df.columns, f"{col} mangler i balance_df"

def test_run_backtest_handles_minimal_input():
    # Prøv med minimal input (fx kun 2 rækker)
    df = make_dummy_df(rows=2)
    trades_df, balance_df = run_backtest(df)
    assert isinstance(trades_df, pd.DataFrame)
    assert isinstance(balance_df, pd.DataFrame)

def test_run_backtest_handles_empty_df():
    # Prøv med tom DataFrame – skal ikke raise error!
    df = pd.DataFrame(columns=[
        "timestamp", "close", "signal", "ema_9", "ema_21", "ema_50", "ema_200",
        "rsi_14", "macd", "macd_signal", "atr_14", "regime"
    ])
    try:
        trades_df, balance_df = run_backtest(df)
        assert isinstance(trades_df, pd.DataFrame)
        assert isinstance(balance_df, pd.DataFrame)
    except Exception as e:
        assert False, f"run_backtest fejlede på tom DataFrame: {e}"

if __name__ == "__main__":
    test_run_backtest_returns_dataframes()
    test_run_backtest_handles_minimal_input()
    test_run_backtest_handles_empty_df()
    print("✅ Alle backtest-tests bestået!")
