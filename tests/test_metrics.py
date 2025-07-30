
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import numpy as np



try:
    from backtest.metrics import calculate_sharpe, calculate_drawdown
except ImportError as e:
    raise ImportError(f"Kunne ikke importere backtest.metrics: {e}")

def test_calculate_sharpe_positive():
    """Sharpe bør være positiv for positive afkast"""
    returns = [0.02, 0.03, 0.01, 0.05]
    sharpe = calculate_sharpe(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0, "Sharpe burde være positiv ved positive returns"

def test_calculate_sharpe_negative():
    """Sharpe kan være negativ for negative afkast"""
    returns = [-0.02, -0.01, -0.03]
    sharpe = calculate_sharpe(returns)
    assert isinstance(sharpe, float)
    assert sharpe < 0, "Sharpe burde være negativ ved negative returns"

def test_calculate_sharpe_zero():
    """Sharpe bør være 0 hvis returns er konstant 0"""
    returns = [0, 0, 0, 0]
    sharpe = calculate_sharpe(returns)
    assert isinstance(sharpe, float)
    assert sharpe == 0, "Sharpe burde være 0 hvis returns er 0"

def test_calculate_drawdown_simple():
    """Drawdown bør være <= 0 og korrekt for et typisk forløb"""
    balance = [100, 120, 80, 130, 110]
    dd = calculate_drawdown(balance)
    assert isinstance(dd, float)
    assert dd <= 0, "Drawdown er altid <= 0"
    # Maksimal drawdown her bør være -0.333... (fra 120 til 80)
    assert np.isclose(dd, -1/3, atol=1e-3)

def test_calculate_drawdown_none():
    """Hvis balancen kun stiger, er drawdown 0"""
    balance = [100, 120, 130, 150]
    dd = calculate_drawdown(balance)
    assert isinstance(dd, float)
    assert dd == 0, "Drawdown skal være 0 hvis balancen aldrig falder"

if __name__ == "__main__":
    test_calculate_sharpe_positive()
    test_calculate_sharpe_negative()
    test_calculate_sharpe_zero()
    test_calculate_drawdown_simple()
    test_calculate_drawdown_none()
    print("✅ Alle metrics-tests bestået!")