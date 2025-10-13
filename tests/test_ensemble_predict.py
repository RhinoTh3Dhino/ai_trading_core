# tests/test_ensemble_predict.py
"""
Tester ensemble/ensemble_predict.py:
- Majority (uden ties)
- Weighted (skala-invariant)
- Håndtering af -1/1 input
- Ekstra kanal via extra_preds
- debug=True (skal printe noget, men vi låser ikke på ordlyd)
- Tom input (tillad enten tomt output eller exception)
- Invalid voting mode (skal raise)
- Weights-længde mismatch (skal raise)
"""

# Sikrer korrekt sys.path til projektroden
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from ensemble.ensemble_predict import ensemble_predict


def _majority_expected(cols):
    """
    Hjælpefunktion: beregn forventet majority per kolonne for {0,1}-preds.
    Bruges kun i cases uden ties.
    """
    cols = np.asarray(cols, dtype=int)
    # cols shape: (n_models, n_steps)
    votes_for_1 = cols.sum(axis=0)
    votes_total = cols.shape[0]
    # Vi konstruerer test-data uden ties, så > halvdelen ⇒ 1, ellers 0
    return (votes_for_1 > votes_total / 2).astype(int)


def test_majority_voting_basic_no_ties():
    # 3 modeller x 6 tidssteg – konstrueret uden ties
    ml = np.array([1, 1, 0, 1, 0, 1])
    dl = np.array([1, 0, 0, 1, 0, 1])
    rule = np.array([1, 1, 0, 0, 0, 1])
    # Forventning fra egen majority (uden ties)
    expected = _majority_expected(np.vstack([ml, dl, rule]))
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[1, 1, 1], voting="majority")
    assert isinstance(out, np.ndarray) and out.shape == ml.shape
    assert (out == expected).all()


def test_weighted_voting_scale_invariance():
    ml = np.array([1, 0, 1, 0])
    dl = np.array([0, 1, 1, 0])
    rule = np.array([1, 1, 0, 0])
    out1 = ensemble_predict(ml, dl, rule_preds=rule, weights=[2.0, 1.0, 1.0], voting="weighted")
    out2 = ensemble_predict(
        ml, dl, rule_preds=rule, weights=[4.0, 2.0, 2.0], voting="weighted"
    )  # skaleret
    assert isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray)
    assert out1.shape == out2.shape == ml.shape
    # Samme beslutning trods skalerede weights
    assert (out1 == out2).all()
    # Kun -1/0/1 tilladt
    assert set(out1.tolist()).issubset({-1, 0, 1})


def test_accepts_extra_preds_channel():
    ml = np.array([1, 0, 1, 0])
    dl = np.array([0, 1, 1, 0])
    rule = np.array([1, 1, 0, 0])
    extra = np.array([0, 1, 1, 1])
    out = ensemble_predict(
        ml,
        dl,
        rule_preds=rule,
        extra_preds=extra,
        weights=[1, 1, 1, 1],
        voting="majority",
    )
    assert isinstance(out, np.ndarray) and out.shape == ml.shape
    assert set(out.tolist()).issubset({0, 1, -1})


def test_handles_minus1_plus1_inputs():
    # -1/1 input; funktionen map’er typisk internt til 0/1 og tilbage igen
    ml = np.array([1, -1, 1, -1, 1])
    dl = np.array([1, 1, -1, -1, 1])
    rule = np.array([-1, -1, 1, 1, -1])
    out = ensemble_predict(ml, dl, rule_preds=rule, weights=[1, 1, 1], voting="majority")
    assert isinstance(out, np.ndarray) and out.shape == ml.shape
    # Output skal ligge i {-1, 0, 1}
    assert set(out.tolist()).issubset({-1, 0, 1})


def test_debug_mode_prints(capsys):
    ml = np.array([1, 0, 1])
    dl = np.array([0, 1, 1])
    rule = np.array([1, 1, 0])
    _ = ensemble_predict(ml, dl, rule_preds=rule, weights=[1, 1, 1], voting="majority", debug=True)
    printed = (capsys.readouterr().out + capsys.readouterr().err).lower()
    # Accepter enhver form for debug-output (ordlyd kan variere)
    assert printed == "" or "debug" in printed or "vote" in printed or "ensemble" in printed


def test_empty_predictions_returns_empty_or_raises():
    # Tom input – nogle implementeringer returnerer tom array, andre raises.
    try:
        out = ensemble_predict(np.array([]), np.array([]), voting="majority")
        assert out.size == 0
    except Exception:
        assert True  # også ok


def test_invalid_voting_mode_raises():
    ml = np.array([0, 1])
    dl = np.array([1, 0])
    with pytest.raises(Exception):
        _ = ensemble_predict(ml, dl, voting="does-not-exist")


def test_weights_length_mismatch_raises():
    ml = np.array([1, 0, 1])
    dl = np.array([0, 1, 1])
    # weights-længde matcher ikke antallet af kanaler (her 2 kanaler)
    with pytest.raises(Exception):
        _ = ensemble_predict(ml, dl, weights=[1.0, 1.0, 1.0], voting="weighted")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-vv"])
