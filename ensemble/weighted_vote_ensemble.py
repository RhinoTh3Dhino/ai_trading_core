import numpy as np


def weighted_vote_ensemble(*signals_list, weights=None, threshold=0.5):
    """
    Ensemble-signal baseret på vægtet sum af valgfrit antal strategier/signaler.
    Parametre:
      - *signals_list: Enhver kombination af signal-arrays/lister (1=BUY, 0=NEUTRAL/HOLD, -1=SELL)
      - weights: array/list/tuple af vægte (samme længde som antal signal-lister). Default: lige vægt.
      - threshold: float. Hvis vægtet sum > threshold → BUY, < -threshold → SELL, ellers 0/NEUTRAL.
    Returnerer: np.ndarray med ensemble-signaler (1=BUY, 0=NEUTRAL, -1=SELL)
    """
    # Konverter inputs til numpy-arrays og check længde
    arrs = [np.asarray(sig, dtype=np.int8) for sig in signals_list]
    n = len(arrs[0])
    if not all(len(a) == n for a in arrs):
        raise ValueError("Alle signal-lister skal have samme længde.")
    n_strat = len(arrs)
    if n_strat < 2:
        raise ValueError("Giv mindst 2 strategier/signallister til ensemble voting.")

    # Brug default weights hvis ikke angivet
    if weights is None:
        weights = np.ones(n_strat)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_strat:
            raise ValueError(
                f"Antal weights ({len(weights)}) matcher ikke antal signal-lister ({n_strat})"
            )

    # Stack signals (shape: (n_strat, n))
    signals = np.vstack(arrs)
    # Vægtet sum pr. bar (axis=0)
    weighted_sum = np.sum(signals * weights.reshape(-1, 1), axis=0)
    # Voting logik
    ensemble_signal = np.where(
        weighted_sum > threshold, 1, np.where(weighted_sum < -threshold, -1, 0)
    ).astype(np.int8)
    return ensemble_signal


# --- Hurtigtest direkte fra terminal ---
if __name__ == "__main__":
    ml = [1, 0, -1, 1, 0]
    rsi = [1, 0, -1, 0, 1]
    macd = [0, 0, 1, -1, -1]
    print("Default weights:", weighted_vote_ensemble(ml, rsi, macd))
    print(
        "Custom weights:",
        weighted_vote_ensemble(ml, rsi, macd, weights=[2, 1, 1], threshold=1.0),
    )
    # Test med fire strategier
    ema = [1, 1, 0, -1, -1]
    print(
        "Fire strategier:",
        weighted_vote_ensemble(ml, rsi, macd, ema, weights=[2, 1, 1, 1]),
    )
