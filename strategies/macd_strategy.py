import numpy as np

def macd_cross_signals(df):
    """
    Returnerer signaler: 1 (BUY), -1 (SELL), 0 (HOLD) baseret på MACD-crossover.
    Forventer kolonnerne 'macd' og 'macd_signal' i df.
    """
    signals = []
    for macd, signal in zip(df['macd'], df['macd_signal']):
        if macd > signal:
            signals.append(1)
        elif macd < signal:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)
