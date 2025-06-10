import numpy as np

def majority_vote_ensemble(*signals_list):
    """
    Returnerer ensemble-signal (1=BUY, -1=SELL, 0=HOLD) baseret på majority voting
    over flere signal-lister (ML, RSI, MACD, ...).
    """
    signals_arr = np.column_stack(signals_list)
    final_signals = []
    for row in signals_arr:
        # Majority voting: hvis flest 1, returner 1, hvis flest -1, returner -1, ellers 0
        vals, counts = np.unique(row, return_counts=True)
        vote = vals[np.argmax(counts)]
        final_signals.append(vote)
    return np.array(final_signals)
