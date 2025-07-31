import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def dummy_returns():
    """Fixture med simple returns for sharpe/drawdown-tests."""
    return [0.01, -0.02, 0.03, 0.02, -0.01]


@pytest.fixture
def dummy_balance():
    """Fixture med balance for drawdown-tests."""
    return [100, 120, 80, 130, 110]


@pytest.fixture
def dummy_features_df():
    """Fixture med et features-DataFrame."""
    df = pd.DataFrame(
        {
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
            "regime": np.random.choice(["bull", "bear"], size=10),
        }
    )
    df.loc[0, "signal"] = 1
    df.loc[1, "signal"] = -1
    return df


@pytest.fixture
def dummy_preds():
    """Fixture til ensemble-voting-tests."""
    return [1, 0, 1, -1, 0, 1]
