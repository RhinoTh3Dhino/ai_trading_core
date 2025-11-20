import numpy as np


def macd_cross_signals(df, macd_col="macd", signal_col="macd_signal", allow_short=True):
    """
    Returnerer signaler: 1 (BUY), -1 (SELL), 0 (HOLD) baseret på MACD-crossover.
    Parametre:
        df: DataFrame med MACD- og signal-kolonner
        macd_col: Navn på MACD-kolonne (default 'macd')
        signal_col: Navn på MACD signal-kolonne (default 'macd_signal')
        allow_short: Hvis False, returner kun 1 (BUY) og 0 (HOLD)
    """
    if macd_col not in df.columns or signal_col not in df.columns:
        raise ValueError(f"Mangler '{macd_col}' eller '{signal_col}' i df: {list(df.columns)}")
    signals = []
    for macd, signal in zip(df[macd_col], df[signal_col]):
        if np.isnan(macd) or np.isnan(signal):
            signals.append(0)
        elif macd > signal:
            signals.append(1)
        elif allow_short and macd < signal:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)
