import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backtest.backtest import run_backtest

def test_run_backtest_returns_dataframes():
    # Dummy-data med alle nødvendige kolonner
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=10, freq="h"),
        "close": np.linspace(100, 110, 10),
        "signal": np.random.choice([1, 0, -1], size=10),
        "ema_9": np.linspace(100, 105, 10),
        "ema_21": np.linspace(100, 108, 10),
        "ema_50": np.linspace(100, 109, 10),
        "ema_200": np.linspace(100, 110, 10),
        "rsi_14": np.random.uniform(30, 70, 10),
        "macd": np.random.uniform(-2, 2, 10),
        "macd_signal": np.random.uniform(-2, 2, 10),
        "atr_14": np.random.uniform(0.5, 2, 10),
        "regime": np.random.choice(["bull", "bear"], size=10)
    })

    trades_df, balance_df = run_backtest(df)
    assert isinstance(trades_df, pd.DataFrame), "trades_df er ikke en DataFrame"
    assert isinstance(balance_df, pd.DataFrame), "balance_df er ikke en DataFrame"
    assert not trades_df.empty, "trades_df er tom"
    assert not balance_df.empty, "balance_df er tom"
