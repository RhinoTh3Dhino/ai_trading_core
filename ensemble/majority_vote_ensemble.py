"""
majority_vote_ensemble.py
-------------------------
Fælles modul til vægtet (eller simpelt) ensemble voting for trading-signaler.
Understøtter både majoritets-voting (uden vægte) og vægtet voting.
Importér fx:
    from ensemble.majority_vote_ensemble import majority_vote_ensemble
    # eller
    from ensemble.majority_vote_ensemble import weighted_vote_ensemble
"""

import numpy as np


def weighted_vote_ensemble(*signals_list, weights=None):
    """
    Ensemble voting med vægte.
    Parametre:
        *signals_list: Flere arrays/lister med signaler fra hver strategi (skal have samme længde).
        weights: Liste/array af vægte for hver strategi. Hvis None, bruges lige vægt (majority).
    Signal-koder:
        -1 = SELL, 0 = HOLD, 1 = BUY
    Returnerer: np.ndarray med ensemble-signaler (-1, 0, 1)
    """
    if not signals_list or any(len(sig) == 0 for sig in signals_list):
        raise ValueError(
            "Alle signal-lister skal være ikke-tomme og have samme længde."
        )
    # Konverter alt til numpy arrays og check længde
    arrs = [np.asarray(sig, dtype=np.int8) for sig in signals_list]
    n = len(arrs[0])
    if not all(len(a) == n for a in arrs):
        raise ValueError("Alle signal-lister skal have samme længde!")
    signals_arr = np.column_stack(arrs)  # shape: (n, k)
    n_strategies = signals_arr.shape[1]

    # Sæt weights til 1 hvis ikke angivet (majority voting)
    if weights is None:
        weights = np.ones(n_strategies)
    else:
        weights = np.array(weights, dtype=float)
        if len(weights) != n_strategies:
            raise ValueError(
                f"Antal weights ({len(weights)}) matcher ikke antal signal-lister ({n_strategies})"
            )

    # Vægtet sum af signaler pr. række
    vote_scores = np.dot(signals_arr, weights)
    # Voting: >0 = BUY, <0 = SELL, 0 = HOLD
    final_signals = np.where(
        vote_scores > 0, 1, np.where(vote_scores < 0, -1, 0)
    ).astype(np.int8)
    return final_signals


# Alias – så begge navne virker
majority_vote_ensemble = weighted_vote_ensemble

# --- Hurtig test ---
if __name__ == "__main__":
    # Tre strategier – test
    ml = np.array([1, -1, 0, 1, -1])
    rsi = np.array([1, -1, 1, 0, 0])
    macd = np.array([1, 0, -1, 0, -1])
    print("Majority (uden vægte):", majority_vote_ensemble(ml, rsi, macd))
    print("Vægtet (2,1,1):", weighted_vote_ensemble(ml, rsi, macd, weights=[2, 1, 1]))
