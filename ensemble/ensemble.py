import numpy as np

def majority_vote_ensemble(*signals):
    """
    Majority voting for vilkårligt antal signal-lister/arrays.
    Returnerer ensemble signal (-1, 0, 1) for hvert datapunkt.
    """
    signals = [np.array(sig) for sig in signals]
    stacked = np.vstack(signals)
    out = []
    for col in stacked.T:
        unique, counts = np.unique(col, return_counts=True)
        max_count = counts.max()
        candidates = unique[counts == max_count]
        if len(candidates) == 1:
            out.append(candidates[0])
        else:
            out.append(0)  # stemmelighed = HOLD
    return np.array(out)
