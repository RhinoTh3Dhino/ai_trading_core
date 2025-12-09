# bot/features/indicators.py

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Eksponentielt glidende gennemsnit.
    Bruges til trend (ema_fast/ema_slow).
    """
    return series.ewm(span=period, adjust=False).mean()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Average True Range (ATR).
    Bruges til volatilitet, SL/TP-beregning og vol_ratio.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()
