import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backtest.backtest import run_backtest

def test_run_backtest_returns_dataframes():
    # Dummy-data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=10, freq="h"),
        "close": np.linspace(100, 110, 10),
        "signal": np.random.choice([1, 0, -1], size=10)
    })
    trades_df, balance_df = run_backtest(df)
    assert isinstance(trades_df, pd.DataFrame), "trades_df er ikke en DataFrame"
    assert isinstance(balance_df, pd.DataFrame), "balance_df er ikke en DataFrame"
    assert not trades_df.empty, "trades_df er tom"
    assert not balance_df.empty, "balance_df er tom"
