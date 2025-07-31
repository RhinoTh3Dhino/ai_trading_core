from utils.project_path import PROJECT_ROOT

# utils/ensemble_utils.py

import json
import os

DEFAULT_ENSEMBLE_THRESHOLD = 0.7
DEFAULT_ENSEMBLE_WEIGHTS = [1.0, 1.0, 0.7]
# AUTO PATH CONVERTED
PARAMS_PATH = PROJECT_ROOT / "models" / "best_ensemble_params.json"


def load_best_ensemble_params():
    """
    Loader threshold og weights til ensemble fra fil, hvis den findes.
    Fallback til standardværdier hvis filen ikke findes eller er korrupt.
    """
    threshold = DEFAULT_ENSEMBLE_THRESHOLD
    weights = DEFAULT_ENSEMBLE_WEIGHTS
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, "r", encoding="utf-8") as f:
                params = json.load(f)
            threshold = params.get("threshold", DEFAULT_ENSEMBLE_THRESHOLD)
            weights = params.get("weights", DEFAULT_ENSEMBLE_WEIGHTS)
            print(f"[ensemble_utils] Loader threshold/weights fra fil: {PARAMS_PATH}")
        except Exception as e:
            print(
                f"[ensemble_utils] Kunne ikke læse {PARAMS_PATH}: {e}. Bruger default."
            )
    else:
        print(f"[ensemble_utils] Ingen {PARAMS_PATH} fundet. Bruger default.")
    return threshold, weights


def save_best_ensemble_params(threshold, weights):
    """
    Gemmer threshold og weights til ensemble i models/best_ensemble_params.json.
    """
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    params = {"threshold": threshold, "weights": weights}
    try:
        with open(PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"[ensemble_utils] Gemte threshold/weights til: {PARAMS_PATH}")
        return True
    except Exception as e:
        print(f"[ensemble_utils] Kunne ikke gemme {PARAMS_PATH}: {e}")
        return False


# (Valgfrit) Her kan du samle flere ensemble/voting-relaterede utils.
# Fx en voting-funktion, ensemble-scorer, analyse osv.


# Eksempel:
def simple_voting(ml_signal, dl_signal, rule_signal, weights=None):
    """
    Simpel weighted voting. Returnerer 1 hvis vægtet sum ≥ 0.5*sum(weights), ellers 0.
    """
    if weights is None:
        weights = DEFAULT_ENSEMBLE_WEIGHTS
    score = weights[0] * ml_signal + weights[1] * dl_signal + weights[2] * rule_signal
    if score >= 0.5 * sum(weights):
        return 1
    return 0
