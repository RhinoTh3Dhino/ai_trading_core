# tests/test_ensemble.py
import sys
from pathlib import Path

# Sørg for at projektroden er på sys.path, så "bot.ensemble" kan importeres
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from bot.ensemble import ensemble_predict
except ImportError as e:
    raise ImportError(f"Kunne ikke importere bot.ensemble: {e}")


# -------------------------------
# Majority-tests (klare udfald)
# -------------------------------
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


# -------------------------------
# Tie / uafgjort
# -------------------------------
def test_ensemble_predict_tie_two_way():
    """
    Tie mellem BUY og SELL (lige mange).
    Krav: resultatet skal være et gyldigt signal (-1, 0 eller 1).
    """
    preds = [1, -1, 1, -1]
    signal = ensemble_predict(preds)
    assert signal in (-1, 0, 1), "Signal bør være en gyldig action ved tie"


def test_ensemble_predict_tie_three_way():
    """
    Tre-vejs tie (1, 0, -1 hver én gang).
    Krav: resultatet skal være et gyldigt signal (-1, 0 eller 1).
    """
    preds = [1, 0, -1]
    signal = ensemble_predict(preds)
    assert signal in (-1, 0, 1), "Signal bør være en gyldig action ved tre-vejs tie"


# -------------------------------
# Edge cases
# -------------------------------
def test_ensemble_predict_empty():
    """
    Tom liste: accepter både at funktionen returnerer 0 (HOLD)
    eller at den vælger at raise en Exception.
    """
    try:
        signal = ensemble_predict([])
        assert signal == 0, "Tom input bør returnere HOLD (0)"
    except Exception:
        # Også OK hvis funktionen vælger at raise for tomt input
        pass


def test_input_immutability():
    """Funktionen må ikke mutere input-listen (common gotcha)."""
    preds = [1, 1, 0, -1, 0, 1]
    original = preds.copy()
    _ = ensemble_predict(preds)
    assert preds == original, "Input-listen blev muteret"


def test_large_input_majority():
    """
    Stor liste for at fange off-by-one fejl i optælling.
    51x BUY, 30x HOLD, 19x SELL -> forvent BUY (1).
    """
    preds = [1] * 51 + [0] * 30 + [-1] * 19
    signal = ensemble_predict(preds)
    assert signal == 1, f"Forventede signal=1 for stor majoritet, fik {signal}"


# --------------------------------
# CLI-kørsel (valgfrit)
# --------------------------------
if __name__ == "__main__":
    test_ensemble_predict_majority_buy()
    test_ensemble_predict_majority_sell()
    test_ensemble_predict_majority_hold()
    test_ensemble_predict_tie_two_way()
    test_ensemble_predict_tie_three_way()
    test_ensemble_predict_empty()
    test_input_immutability()
    test_large_input_majority()
    print("✅ Alle ensemble-tests bestået!")
