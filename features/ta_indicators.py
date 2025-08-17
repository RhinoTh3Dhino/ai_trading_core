import pandas as pd
import pandas_ta as ta
import numpy as np

# Sikrer kompatibilitet med nyere NumPy-versioner
npNaN = np.nan


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sørger for at DataFrame har en DatetimeIndex (kræves af fx VWAP).
    Hvis der ikke er timestamp-kolonne, oprettes en kunstig dato-range.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"]))
        else:
            df = df.set_index(pd.date_range(start="2000-01-01", periods=len(df), freq="D"))
    return df


def add_ta_indicators(df: pd.DataFrame, force_no_supertrend: bool = False) -> pd.DataFrame:
    """
    Tilføjer tekniske indikatorer til et DataFrame.
    force_no_supertrend=True simulerer en fejl i supertrend-beregning (til test).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input skal være en pandas DataFrame")

    required_cols = ["close", "high", "low", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner: {missing}")

    df = df.copy()
    df = _ensure_datetime_index(df)

    # EMA
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_200"] = ta.ema(df["close"], length=200)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_50"] = ta.ema(df["close"], length=50)

    # MACD
    macd = ta.macd(df["close"])
    if macd is not None and not macd.empty:
        df["macd"] = macd.get("MACD_12_26_9", npNaN)
        df["macd_signal"] = macd.get("MACDs_12_26_9", npNaN)
        df["macd_hist"] = macd.get("MACDh_12_26_9", npNaN)
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = npNaN

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_28"] = ta.rsi(df["close"], length=28)

    # ATR
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20)
    if bbands is not None and not bbands.empty:
        df["bb_upper"] = bbands.get("BBU_20_2.0", npNaN)
        df["bb_middle"] = bbands.get("BBM_20_2.0", npNaN)
        df["bb_lower"] = bbands.get("BBL_20_2.0", npNaN)
    else:
        df["bb_upper"] = df["bb_middle"] = df["bb_lower"] = npNaN

    # VWAP – kræver datetime index
    try:
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    except Exception as e:
        print(f"[WARN] VWAP kunne ikke beregnes: {e}")
        df["vwap"] = npNaN

    # OBV
    df["obv"] = ta.obv(df["close"], df["volume"])

    # ADX
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = adx_df["ADX_14"] if adx_df is not None and "ADX_14" in adx_df else npNaN

    # Z-score
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()

    # Supertrend (med mulighed for at simulere fejl)
    try:
        if force_no_supertrend:
            raise RuntimeError("Simuleret supertrend-fejl")
        st = ta.supertrend(df["high"], df["low"], df["close"])
        df["supertrend"] = st["SUPERT_7_3.0"] if st is not None and "SUPERT_7_3.0" in st else npNaN
    except Exception:
        df["supertrend"] = npNaN

    # Volume spike
    df["volume_spike"] = df["volume"] > df["volume"].rolling(20).mean() * 1.5

    # Regime: Bull hvis ema_9 > ema_21
    df["regime"] = (df["ema_9"] > df["ema_21"]).astype(int)

    # Drop NaN (kun på indikatorer, ikke nødvendigvis hele dataset)
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Simpel selvtest (dummy-data)
    test_df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=50, freq="D"),
        "close": np.random.rand(50) * 100,
        "high": np.random.rand(50) * 100,
        "low": np.random.rand(50) * 100,
        "volume": np.random.rand(50) * 1000
    })
    print(add_ta_indicators(test_df).head())
