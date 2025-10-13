# features/patterns.py
"""
Pattern-baserede features til AI trading pipeline:
- Breakout detection (op/ned)
- Volume spike
- Simple candlestick patterns (bullish/bearish engulfing, doji, hammer)
"""

import numpy as np
import pandas as pd


def add_breakout_and_volume_spike(df, lookback=20, vol_mult=2.0):
    """
    Tilføjer breakout og volume spike features til DataFrame.
    - breakout_up: Pris lukker over seneste high (lookback)
    - breakout_down: Pris lukker under seneste low (lookback)
    - vol_spike: Volume er > X gange gennemsnittet (lookback)
    """
    df = df.copy()
    df["breakout_up"] = (df["close"] > df["high"].rolling(lookback).max().shift(1)).astype(int)
    df["breakout_down"] = (df["close"] < df["low"].rolling(lookback).min().shift(1)).astype(int)
    df["vol_spike"] = (df["volume"] > df["volume"].rolling(lookback).mean() * vol_mult).astype(int)
    return df


def add_candlestick_patterns(df):
    """
    Tilføjer simple candlestick-mønstre (engulfing, doji, hammer) som binære features.
    """
    df = df.copy()

    # Bullish Engulfing
    df["bull_engulf"] = (
        (df["close"].shift(1) < df["open"].shift(1))  # Forrige rød
        & (df["close"] > df["open"])  # Nu grøn
        & (df["close"] > df["open"].shift(1))
        & (df["open"] < df["close"].shift(1))
    ).astype(int)

    # Bearish Engulfing
    df["bear_engulf"] = (
        (df["close"].shift(1) > df["open"].shift(1))  # Forrige grøn
        & (df["close"] < df["open"])  # Nu rød
        & (df["close"] < df["open"].shift(1))
        & (df["open"] > df["close"].shift(1))
    ).astype(int)

    # Doji
    df["doji"] = (np.abs(df["close"] - df["open"]) < (df["high"] - df["low"]) * 0.1).astype(int)

    # Hammer (simpel definition)
    body = np.abs(df["close"] - df["open"])
    lower_shadow = df["open"].combine(df["close"], min) - df["low"]
    upper_shadow = df["high"] - df["open"].combine(df["close"], max)
    df["hammer"] = ((lower_shadow > body * 2) & (upper_shadow < body)).astype(int)

    return df


# === Samlet convenience-funktion ===
def add_all_patterns(df, breakout_lookback=20, vol_mult=2.0):
    """
    Tilføj alle relevante pattern-features til DataFrame.
    """
    df = add_breakout_and_volume_spike(df, lookback=breakout_lookback, vol_mult=vol_mult)
    df = add_candlestick_patterns(df)
    return df
