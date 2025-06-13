import numpy as np

def weighted_vote_ensemble(*signals_list, weights=None):
    """
    Ensemble voting med vægte.
    Eksempel:
        weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, weights=[1.0, 0.8, 0.6])
    Hvis weights=None, køres simpelt majority voting (alle får vægt 1).
    """
    signals_arr = np.column_stack(signals_list)  # shape: (n, k)
    n_strategies = signals_arr.shape[1]

    # Sæt weights til 1 hvis ikke angivet (majority voting)
    if weights is None:
        weights = [1.0] * n_strategies
    weights = np.array(weights)

    # Vægtet sum af signaler pr. række
    vote_scores = np.dot(signals_arr, weights)
    final_signals = np.where(vote_scores > 0, 1, np.where(vote_scores < 0, -1, 0))
    return final_signals

# Alias – så du kan importere og bruge majority_vote_ensemble præcis som før
majority_vote_ensemble = weighted_vote_ensemble
