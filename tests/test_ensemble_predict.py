# tests/test_ensemble_predict.py

import sys
import os
import numpy as np

# Tilføj projektroden til sys.path, så ensemble/ kan importeres korrekt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ensemble.ensemble_predict import ensemble_predict

def test_majority_voting_basic():
    ml = np.array([0, 1, 1, 0, 1])
    dl = np.array([1, 1, 0, 0, 1])
    rule = np.array([0, 1, 1, 1, 0])
    # Majority voting: vægt 1 til alle
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[1,1,1], voting="majority")
    # Forventet output: 0, 1, 1, 0, 1 (fordi >1 af 3 stemmer på 1)
    assert (out == np.array([0, 1, 1, 0, 1])).all()

def test_weighted_voting():
    ml = np.array([1, 0, 1, 0])
    dl = np.array([0, 1, 1, 0])
    rule = np.array([1, 1, 0, 0])
    # Weighted voting: stærk vægt på ML
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[2,1,1], voting="weighted")
    # Når ML har vægt 2, så burde output følge MLs flertal hvis der er tvivl
    assert (out == np.array([1, 0, 1, 0])).all()

def test_debug_mode_prints(capsys):
    ml = np.array([1, 0, 1])
    dl = np.array([0, 1, 1])
    rule = np.array([1, 1, 0])
    # Aktiver debug
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[1,1,1], voting="majority", debug=True)
    # Tjek at der bliver printet noget med "[Ensemble]"
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
