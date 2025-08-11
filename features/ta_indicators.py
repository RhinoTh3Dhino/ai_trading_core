import pandas as pd
import pandas_ta as ta
import numpy as np

# Sikrer kompatibilitet med nyere NumPy-versioner
npNaN = np.nan


def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_200"] = ta.ema(df["close"], length=200)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_50"] = ta.ema(df["close"], length=50)

    # MACD
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_28"] = ta.rsi(df["close"], length=28)

    # ATR
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20)
    df["bb_upper"] = bbands["BBU_20_2.0"]
    df["bb_middle"] = bbands["BBM_20_2.0"]
    df["bb_lower"] = bbands["BBL_20_2.0"]

    # VWAP
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

    # OBV
    df["obv"] = ta.obv(df["close"], df["volume"])

    # ADX
    df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # Z-score
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()

    # Supertrend (hvis du har pandas_ta >= 0.3.14)
    try:
        st = ta.supertrend(df["high"], df["low"], df["close"])
        df["supertrend"] = st["SUPERT_7_3.0"]
    except Exception:
        df["supertrend"] = npNaN

    # Volume spike
    df["volume_spike"] = df["volume"] > df["volume"].rolling(20).mean() * 1.5

    # Regime: Bull hvis ema_9 > ema_21
    df["regime"] = (df["ema_9"] > df["ema_21"]).astype(int)

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    return df
