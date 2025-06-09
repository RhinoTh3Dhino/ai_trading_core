def rsi_rule_based_signals(df, low=30, high=70):
    """
    Returnerer -1 (SELL), 0 (HOLD), 1 (BUY) baseret på RSI threshold.
    Forventer at df indeholder kolonnen 'rsi_14'.
    """
    signals = []
    for val in df['rsi_14']:
        if val < low:
            signals.append(1)
        elif val > high:
            signals.append(-1)
        else:
            signals.append(0)
    return signals
