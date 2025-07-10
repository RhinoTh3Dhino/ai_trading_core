import numpy as np

def rsi_rule_based_signals(
    df, 
    rsi_col="rsi_14", 
    low=30, 
    high=70, 
    allow_short=True
):
    """
    Returnerer signaler baseret på RSI:
    -1 (SELL), 0 (HOLD), 1 (BUY)
    Parametre:
        df: DataFrame med RSI-kolonne
        rsi_col: Navn på RSI-kolonne (default 'rsi_14')
        low: Køb-threshold (default 30)
        high: Salg-threshold (default 70)
        allow_short: Hvis False: kun 1 eller 0 signaler (ingen -1 SELL)
    """
    if rsi_col not in df.columns:
        raise ValueError(f"Kolonnen '{rsi_col}' mangler i df: {list(df.columns)}")
    signals = []
    for val in df[rsi_col]:
        if np.isnan(val):
            signals.append(0)  # HOLD hvis NaN
        elif val < low:
            signals.append(1)
        elif allow_short and val > high:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)
