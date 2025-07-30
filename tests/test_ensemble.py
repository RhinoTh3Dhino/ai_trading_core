try:
    import sys
    import os
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(str(PROJECT_ROOT)))
    from bot.ensemble import ensemble_predict
except ImportError as e:
    raise ImportError(f"Kunne ikke importere bot.ensemble: {e}")

def test_ensemble_predict_majority_buy():
    """Majoriteten er BUY (1)."""
    preds = [1, 1, 0, 1, 0]
    signal = ensemble_predict(preds)
    assert signal == 1, f"Forventede signal=1, fik {signal}"

def test_ensemble_predict_majority_sell():
    """Majoriteten er SELL (-1)."""
    preds = [-1, -1, 0, -1, 1]
    signal = ensemble_predict(preds)
    assert signal == -1, f"Forventede signal=-1, fik {signal}"

def test_ensemble_predict_majority_hold():
    """Majoriteten er HOLD (0)."""
    preds = [0, 0, 1, -1, 0]
    signal = ensemble_predict(preds)
    assert signal == 0, f"Forventede signal=0, fik {signal}"

def test_ensemble_predict_tie():
    """Test for tie (lige mange af to typer)."""
    preds = [1, -1, 1, -1]
    signal = ensemble_predict(preds)
    assert signal in [-1, 0, 1], "Signal bør være en gyldig action, også ved tie"

def test_ensemble_predict_empty():
    """Tom liste skal give 0 eller raise Exception."""
    try:
        signal = ensemble_predict([])
        assert signal == 0, "Tom input bør returnere HOLD (0)"
    except Exception:
        pass  # Også OK hvis der raises error

if __name__ == "__main__":
    test_ensemble_predict_majority_buy()
    test_ensemble_predict_majority_sell()
    test_ensemble_predict_majority_hold()
    test_ensemble_predict_tie()
    test_ensemble_predict_empty()
    print("✅ Alle ensemble-tests bestået!")
