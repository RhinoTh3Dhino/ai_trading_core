import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Beregn Relative Strength Index (RSI) for givet periode."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)  # Undgå division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def add_ema(df: pd.DataFrame, span: int, col_name: str = "ema_") -> pd.DataFrame:
    """Tilføj eksponentiel glidende gennemsnit (EMA) som ny kolonne."""
    df[f"{col_name}{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df


def add_sma(df: pd.DataFrame, window: int, col_name: str = "sma_") -> pd.DataFrame:
    """Tilføj simpelt glidende gennemsnit (SMA) som ny kolonne."""
    df[f"{col_name}{window}"] = df["close"].rolling(window=window).mean()
    return df


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn MACD, MACD-signal og MACD-histogram."""
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Beregn Average True Range (ATR) for givet periode."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.fillna(0)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Beregn Average Directional Index (ADX)."""
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff(-1)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift())
    tr3 = abs(df["low"] - df["close"].shift())
    tr = np.max([tr1, tr2, tr3], axis=0)
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).sum() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).sum() / (atr + 1e-9)
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9) * 100
    adx = pd.Series(dx).rolling(window=period).mean()
    return pd.Series(adx, index=df.index).fillna(0)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Beregn On Balance Volume (OBV)."""
    obv = [0]
    for i in range(1, len(df)):
        if df.loc[i, "close"] > df.loc[i - 1, "close"]:
            obv.append(obv[-1] + df.loc[i, "volume"])
        elif df.loc[i, "close"] < df.loc[i - 1, "close"]:
            obv.append(obv[-1] - df.loc[i, "volume"])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def calculate_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0):
    """Beregn Bollinger Bands."""
    rolling_mean = df["close"].rolling(window=window).mean()
    rolling_std = df["close"].rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper, lower


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Beregn Commodity Channel Index (CCI)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - ma) / (0.015 * md + 1e-9)
    return cci.fillna(0)


def calculate_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    """Beregn Supertrend."""
    atr = calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    supertrend = [True]
    for i in range(1, len(df)):
        curr_close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        if curr_close > final_upperband.iloc[i - 1]:
            supertrend.append(True)
        elif curr_close < final_lowerband.iloc[i - 1]:
            supertrend.append(False)
        else:
            supertrend.append(supertrend[-1])
        if supertrend[-1]:
            final_lowerband.iloc[i] = max(
                lowerband.iloc[i], final_lowerband.iloc[i - 1]
            )
        else:
            final_upperband.iloc[i] = min(
                upperband.iloc[i], final_upperband.iloc[i - 1]
            )
    return pd.Series(supertrend, index=df.index).astype(int)


# Udvid let med flere indikatorer, fx momentum, volume-anomali osv.
