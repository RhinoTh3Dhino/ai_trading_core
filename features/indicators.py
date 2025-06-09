import pandas as pd

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Beregn Relative Strength Index (RSI) for givet periode."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def add_ema(df: pd.DataFrame, span: int, col_name: str = 'ema_') -> pd.DataFrame:
    """Tilføj eksponentiel glidende gennemsnit (EMA) som ny kolonne."""
    df[f'{col_name}{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df

def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn MACD og MACD-signal for DataFrame."""
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Beregn Average True Range (ATR) for givet periode."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.fillna(0)

# Du kan nemt tilføje flere indikatorer her, fx SMA, CCI, ADX mv.
