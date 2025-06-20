import numpy as np

def weighted_vote_ensemble(ml_signals, rsi_signals, macd_signals, weights=None):
    """
    Ensemble-signal baseret på vægtet sum af strategier.
    - ml_signals, rsi_signals, macd_signals: arrays eller lists med signaler (1=BUY, 0=NEUTRAL, -1=SELL)
    - weights: list/tuple/array af vægte, fx [1.0, 0.7, 0.4]
    Returnerer: array med ensemble-signaler (1=BUY, 0=NEUTRAL, -1=SELL)
    """
    ml_signals = np.array(ml_signals)
    rsi_signals = np.array(rsi_signals)
    macd_signals = np.array(macd_signals)
    signals = np.vstack([ml_signals, rsi_signals, macd_signals])
    if weights is None:
        weights = [1.0, 0.7, 0.4]
    weights = np.array(weights).reshape(-1, 1)
    # Vægtet sum
    weighted_sum = np.sum(signals * weights, axis=0)
    # Thresholds: >0.5 = BUY, <-0.5 = SELL, ellers 0 (NEUTRAL)
    ensemble_signal = np.where(weighted_sum > 0.5, 1,
                        np.where(weighted_sum < -0.5, -1, 0))
    return ensemble_signal
