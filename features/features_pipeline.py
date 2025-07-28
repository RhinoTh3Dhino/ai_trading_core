# features/features_pipeline.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from utils.project_path import PROJECT_ROOT

# Importér patterns direkte
from features.patterns import add_all_patterns

def generate_features(df: pd.DataFrame, feature_config: dict = None) -> pd.DataFrame:
    """Samlet pipeline til at beregne tekniske indikatorer + pattern-features."""

    df = df.copy()

    # --- Kolonne-mapping (datetime/timestamp) ---
    if 'datetime' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    if 'Timestamp' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'Timestamp': 'timestamp'})

    # --- Tjek basis-kolonner ---
    required_cols = ['timestamp', 'close', 'volume', 'open', 'high', 'low']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner: {missing} (fandt: {list(df.columns)})")

    # --- Konverter til datetime & sortér ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- Konverter til numeric ---
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # === TEKNISKE INDIKATORER ===
    # EMA 9/21/50
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    # RSI 14 og 28
    for rsi_n in [14, 28]:
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(rsi_n).mean()
        avg_loss = pd.Series(loss).rolling(rsi_n).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f'rsi_{rsi_n}'] = 100 - (100 / (1 + rs))
    # MACD & Signal
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    # ATR 14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    # Return og PV-ratio
    df['return'] = df['close'].pct_change().fillna(0)
    df['pv_ratio'] = df['close'] / (df['volume'] + 1e-9)
    # Regime (bull hvis ema_9 > ema_21)
    df['regime'] = (df['ema_9'] > df['ema_21']).astype(int)
    # BOLLINGER BANDS (20 perioder, 2 std)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # === PATTERN & BREAKOUT FEATURES ===
    df = add_all_patterns(df, breakout_lookback=20, vol_mult=2.0)

    # --- ENSURE volume_spike-kolonne eksisterer ---
    # Hvis add_all_patterns genererer 'vol_spike', omdøb den til 'volume_spike'
    if "vol_spike" in df.columns:
        df.rename(columns={"vol_spike": "volume_spike"}, inplace=True)

    # --- Fjern rækker med NaN i tekniske features ---
    feature_cols = [
        'rsi_14', 'rsi_28', 'ema_9', 'ema_21', 'ema_50', 'macd', 'macd_signal', 'vwap', 'atr_14', 'regime',
        'bb_upper', 'bb_lower',
        'breakout_up', 'breakout_down', 'volume_spike', 'bull_engulf', 'bear_engulf', 'doji', 'hammer'
    ]
    # Kun drop kolonner der eksisterer i df
    df = df.dropna(subset=[col for col in feature_cols if col in df.columns])

    # --- Tilføj target hvis ikke findes (fallback) ---
    if not any([col for col in df.columns if str(col).startswith('target')]):
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.reset_index(drop=True, inplace=True)
    return df

def save_features(df: pd.DataFrame, symbol: str, timeframe: str, version: str = "v1") -> str:
    today = datetime.now().strftime('%Y%m%d')
    filename = f"{symbol.lower()}_{timeframe}_features_{version}_{today}.csv"
    # Brug Pathlib overalt
    output_dir = Path(PROJECT_ROOT) / "outputs" / "feature_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / filename
    df.to_csv(full_path, index=False)
    print(f"✅ Features gemt: {full_path}")
    return str(full_path)

def load_features(symbol: str, timeframe: str, version_prefix: str = "v1") -> pd.DataFrame:
    folder = Path(PROJECT_ROOT) / "outputs" / "feature_data"
    # Brug Path-API så det virker både på Windows og Linux
    files = [f for f in folder.iterdir() if f.is_file() and str(f).startswith(f"{symbol.lower()}_{timeframe}_features_{version_prefix}")]
    if not files:
        raise FileNotFoundError(f"Ingen feature-filer fundet for {symbol} {timeframe} ({version_prefix})")
    files.sort(key=lambda x: x.name, reverse=True)
    newest_file = files[0]
    print(f"📥 Indlæser features: {newest_file}")
    return pd.read_csv(newest_file)

if __name__ == "__main__":
    # Simpel test på en dummy-fil
    path = Path(PROJECT_ROOT) / "data" / "BTCUSDT_1h_with_target.csv"
    if path.exists():
        df = pd.read_csv(path)
        features = generate_features(df)
        print("Shape:", features.shape)
        print("Kolonner:", features.columns)
        print(features.head())
    else:
        print(f"Testfil mangler: {path}")
