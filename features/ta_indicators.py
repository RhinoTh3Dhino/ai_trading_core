# features/ta_indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd
import ta  # https://github.com/bukosabino/ta

npNaN = np.nan


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sørger for at DataFrame har en DatetimeIndex (nogle indikatorer/VWAP har glæde af det).
    Hvis der ikke er en 'timestamp'-kolonne, oprettes en kunstig dato-range.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"]))
        else:
            df = df.set_index(
                pd.date_range(start="2000-01-01", periods=len(df), freq="D")
            )
    return df


def add_ta_indicators(
    df: pd.DataFrame, force_no_supertrend: bool = False
) -> pd.DataFrame:
    """
    Tilføjer tekniske indikatorer til et DataFrame vha. 'ta'-biblioteket (ikke pandas_ta).
    - EMA(9/21/50/200), MACD, RSI(14/28), ATR(14), Bollinger(20,2),
      VWAP(typisk 14), OBV, ADX(14), Z-score(20), volume_spike, regime.
    - 'Supertrend' findes ikke i 'ta' og sættes derfor til NaN (med mulighed for
      simuleret fejl via force_no_supertrend).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input skal være en pandas DataFrame")

    required = ["close", "high", "low", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner: {missing}")

    df = df.copy()
    df = _ensure_datetime_index(df)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # EMA'er
    df["ema_9"] = ta.trend.EMAIndicator(
        close=close, window=9, fillna=False
    ).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(
        close=close, window=21, fillna=False
    ).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(
        close=close, window=50, fillna=False
    ).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(
        close=close, window=200, fillna=False
    ).ema_indicator()

    # MACD
    macd_ind = ta.trend.MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9, fillna=False
    )
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14, fillna=False).rsi()
    df["rsi_28"] = ta.momentum.RSIIndicator(close=close, window=28, fillna=False).rsi()

    # ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14, fillna=False
    ).average_true_range()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(
        close=close, window=20, window_dev=2, fillna=False
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # VWAP (14 som default-vindue i 'ta')
    try:
        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=vol, window=14, fillna=False
        )
        df["vwap"] = vwap.volume_weighted_average_price()
    except Exception as e:
        print(f"[WARN] VWAP kunne ikke beregnes: {e}")
        df["vwap"] = npNaN

    # OBV
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=vol, fillna=False
    ).on_balance_volume()

    # ADX
    df["adx_14"] = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=14, fillna=False
    ).adx()

    # Z-score (20)
    roll = close.rolling(20)
    df["zscore_20"] = (close - roll.mean()) / roll.std()

    # "Supertrend" ikke i 'ta' -> sæt NaN (og tillad simuleret fejl)
    try:
        if force_no_supertrend:
            raise RuntimeError("Simuleret supertrend-fejl")
        # Ingen native supertrend i 'ta'; behold NaN-kolonne:
        df["supertrend"] = npNaN
    except Exception:
        df["supertrend"] = npNaN

    # Volume spike (enkelt heuristik)
    df["volume_spike"] = vol > vol.rolling(20).mean() * 1.5

    # Regime: Bull hvis ema_9 > ema_21
    df["regime"] = (df["ema_9"] > df["ema_21"]).astype(int)

    # Drop rækker med NaN i indikatorer (valgfrit – beholder kun "modne" rækker)
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Hurtig selvtest
    n = 200
    test_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="H"),
            "close": np.linspace(100, 120, n) + np.random.randn(n) * 2,
            "high": np.linspace(101, 121, n) + np.random.randn(n) * 2,
            "low": np.linspace(99, 119, n) + np.random.randn(n) * 2,
            "volume": np.abs(np.random.randn(n) * 1000) + 10,
        }
    )
    out = add_ta_indicators(test_df)
    print(out.head())
