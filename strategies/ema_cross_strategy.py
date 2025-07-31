import numpy as np


def ema_cross_signals(df, fast_col="ema_9", slow_col="ema_21", allow_short=True):
    """
    Returnerer signaler: 1 (BUY), -1 (SELL), 0 (HOLD) baseret på EMA-crossover.
    Parametre:
        df: DataFrame med EMA-kolonner
        fast_col: Navn på hurtig EMA (default 'ema_9')
        slow_col: Navn på langsom EMA (default 'ema_21')
        allow_short: Hvis False, returneres kun 1 (BUY) og 0 (HOLD)
    """
    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(
            f"Mangler '{fast_col}' eller '{slow_col}' i df: {list(df.columns)}"
        )
    signals = []
    for fast, slow in zip(df[fast_col], df[slow_col]):
        if np.isnan(fast) or np.isnan(slow):
            signals.append(0)
        elif fast > slow:
            signals.append(1)
        elif allow_short and fast < slow:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)


# Eksempel på brug i engine eller backtest:
# signals = ema_cross_signals(df, fast_col="ema_9", slow_col="ema_21", allow_short=True)
