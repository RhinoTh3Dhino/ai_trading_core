"""
tests/test_ensemble_predict.py

Tester ensemble_predict funktionalitet.
"""

# Sikrer korrekt sys.path til projektroden
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
import os
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent  # AUTO-FIXED PATHLIB
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ensemble.ensemble_predict import ensemble_predict

def test_majority_voting_basic():
    ml = np.array([0, 1, 1, 0, 1])
    dl = np.array([1, 1, 0, 0, 1])
    rule = np.array([0, 1, 1, 1, 0])
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[1,1,1], voting="majority")
    assert (out == np.array([0, 1, 1, 0, 1])).all()

def test_weighted_voting():
    ml = np.array([1, 0, 1, 0])
    dl = np.array([0, 1, 1, 0])
    rule = np.array([1, 1, 0, 0])
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[2,1,1], voting="weighted")
    assert (out == np.array([1, 0, 1, 0])).all()

def test_debug_mode_prints(capsys):
    ml = np.array([1, 0, 1])
    dl = np.array([0, 1, 1])
    rule = np.array([1, 1, 0])
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[1,1,1], voting="majority", debug=True)
    captured = capsys.readouterr()
    assert "[Ensemble]" in captured.out

def test_error_on_invalid_voting():
    ml = np.array([0,1])
    dl = np.array([1,0])
    try:
        ensemble_predict(ml, dl, voting="invalid")
        assert False, "Skulle have rejst ValueError"
    except ValueError as e:
        assert "Ukendt voting-type" in str(e)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])