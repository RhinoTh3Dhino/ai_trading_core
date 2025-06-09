import os
import re
import pandas as pd
from datetime import datetime
from features.indicators import (
    calculate_rsi, calculate_macd, add_ema, calculate_atr
)
from features.preprocessing import normalize_zscore

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Samlet pipeline til at beregne alle tekniske indikatorer og forberede feature-matrix."""

    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # RSI (14 og 28)
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['rsi_28'] = calculate_rsi(df, period=28)

    # MACD + signal
    df = calculate_macd(df)

    # EMA: 9, 21, 50, 200
    for span in [9, 21, 50, 200]:
        df = add_ema(df, span=span, col_name='ema_')

    # ATR (volatilitet, SL/TP)
    df['atr_14'] = calculate_atr(df, period=14)

    # VWAP (volume-weighted average price)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Bollinger Bands (over/under EMA50 ± std)
    df['bb_upper'] = df['ema_50'] + df['close'].rolling(window=20).std()
    df['bb_lower'] = df['ema_50'] - df['close'].rolling(window=20).std()

    # Return (momentum) og price/volume ratio
    df['return'] = df['close'].pct_change().fillna(0)
    df['pv_ratio'] = df['close'] / (df['volume'] + 1e-9)

    # Volume spike (anomaly flag)
    df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5

    # Regime: Bull hvis ema_9 > ema_21, ellers Bear (kan udvides)
    df['regime'] = (df['ema_9'] > df['ema_21']).astype(int)

    # Z-score normalisering på centrale features
    feature_cols = [
        'rsi_14', 'rsi_28', 'macd', 'macd_signal', 'ema_9', 'ema_21', 'ema_50',
        'atr_14', 'return', 'pv_ratio'
    ]
    df = normalize_zscore(df, feature_cols)

    # Drop NaN fra starten (rolling window effekt)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def save_features(df: pd.DataFrame, symbol: str, timeframe: str, version: str = "v1") -> str:
    """Gemmer feature-matrix med intelligent navn i outputs/feature_data/."""
    today = datetime.now().strftime('%Y%m%d')
    filename = f"{symbol.lower()}_{timeframe}_features_{version}_{today}.csv"
    output_dir = "outputs/feature_data/"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"✅ Features gemt: {full_path}")
    return full_path

def load_features(symbol: str, timeframe: str, version_prefix: str = "v1") -> pd.DataFrame:
    """Indlæser nyeste version af feature-matrix for en given coin og timeframe."""
    folder = "outputs/feature_data/"
    pattern = re.compile(rf"{symbol.lower()}_{timeframe}_features_{version_prefix}.*\.csv")
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"Ingen feature-filer fundet for {symbol} {timeframe} ({version_prefix})")
    # Sortér efter dato i filnavnet YYYYMMDD
    files.sort(key=lambda x: re.findall(r"_(\d{8})\.csv", x)[-1], reverse=True)
    newest_file = os.path.join(folder, files[0])
    print(f"📥 Indlæser features: {newest_file}")
    return pd.read_csv(newest_file)

# Eksempel på brug (fx i engine.py eller notebook):
if __name__ == "__main__":
    # Tilpas denne path til din rå-data!
    raw_df = pd.read_csv("outputs/data/btc_1h_raw.csv")
    features = generate_features(raw_df)
    save_features(features, symbol="BTC", timeframe="1h", version="v1.3")
