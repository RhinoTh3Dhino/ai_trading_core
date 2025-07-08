import sys
import os
import re
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importér konfiguration centralt
from config.config import FEATURES, COINS, TIMEFRAMES

from features.indicators import (
    calculate_rsi, calculate_macd, add_ema, calculate_atr
)
from features.preprocessing import normalize_zscore

def generate_features(df: pd.DataFrame, feature_config: dict = None) -> pd.DataFrame:
    """Samlet pipeline til at beregne tekniske indikatorer ud fra config.py."""

    df = df.copy()

    # --- Automatisk kolonne-mapping (datetime/timestamp) ---
    if 'datetime' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    if 'Timestamp' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'Timestamp': 'timestamp'})

    # --- Robust tjek af basiskolonner ---
    required_cols = ['timestamp', 'close', 'volume', 'open', 'high', 'low']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner: {missing} (fandt: {list(df.columns)})")

    # --- Konverter til datetime & sortér korrekt ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- Konverter til numeric (robusthed mod teksttal fra CSV) ---
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Brug config eller default
    features = feature_config if feature_config else FEATURES

    # Trend: EMA, MACD
    for span in features.get("trend", []):
        if "ema" in span:  # fx 'ema_21'
            span_num = int(span.split("_")[1])
            df = add_ema(df, span=span_num)
    if "macd" in features.get("trend", []):
        df = calculate_macd(df)

    # Momentum: RSI
    for rsi_str in features.get("momentum", []):
        if "rsi" in rsi_str:
            rsi_num = int(rsi_str.split("_")[1])
            df[f"rsi_{rsi_num}"] = calculate_rsi(df, period=rsi_num)

    # Volatility: ATR, Bollinger
    if "atr_14" in features.get("volatility", []):
        df["atr_14"] = calculate_atr(df, period=14)
    if "bb_upper" in features.get("volatility", []) and "ema_50" in df.columns:
        rolling_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["ema_50"] + rolling_std
        df["bb_lower"] = df["ema_50"] - rolling_std

    # Volume: VWAP
    if "vwap" in features.get("volume", []):
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')
        df["vwap"] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df = df.reset_index()

    # Ekstra: Z-score, regime
    if "zscore_20" in features.get("regime", []):
        df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    if "adx_14" in features.get("regime", []):
        pass  # Implementér evt. ADX

    # Momentum/return
    df['return'] = df['close'].pct_change().fillna(0)
    df['pv_ratio'] = df['close'] / (df['volume'] + 1e-9)
    df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5

    # Regime-label: Bull hvis ema_9 > ema_21
    if all(col in df.columns for col in ["ema_9", "ema_21"]):
        df['regime'] = (df['ema_9'] > df['ema_21']).astype(int)

    # Z-score normalisering på centrale features
    feature_cols = []
    for group in features.values():
        feature_cols += [f for f in group if f in df.columns]
    feature_cols = list(set(feature_cols))  # Unique
    if feature_cols:
        df = normalize_zscore(df, feature_cols)

    # --- NYT: Tilføj target-kolonne ---
    # Du kan justere logikken hvis du har flere klasser
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    # Eller hvis du bruger 3-klasser:
    # df["target"] = df["close"].shift(-1) - df["close"]
    # df["target"] = df["target"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def save_features(df: pd.DataFrame, symbol: str, timeframe: str, version: str = "v1") -> str:
    today = datetime.now().strftime('%Y%m%d')
    filename = f"{symbol.lower()}_{timeframe}_features_{version}_{today}.csv"
    output_dir = "outputs/feature_data/"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"✅ Features gemt: {full_path}")
    return full_path

def load_features(symbol: str, timeframe: str, version_prefix: str = "v1") -> pd.DataFrame:
    folder = "outputs/feature_data/"
    pattern = re.compile(rf"{symbol.lower()}_{timeframe}_features_{version_prefix}.*\.csv")
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"Ingen feature-filer fundet for {symbol} {timeframe} ({version_prefix})")
    files.sort(key=lambda x: re.findall(r"_(\d{8})\.csv", x)[-1], reverse=True)
    newest_file = os.path.join(folder, files[0])
    print(f"📥 Indlæser features: {newest_file}")
    return pd.read_csv(newest_file)

def test_pipeline():
    """Kør automatisk test af feature-pipeline på én testfil."""
    test_path = "outputs/data/btcusdt_1h_raw.csv"
    if not os.path.exists(test_path):
        print(f"❌ Testfil mangler: {test_path}")
        return
    df = pd.read_csv(test_path)
    features = generate_features(df)
    print("✅ Feature-matrix shape:", features.shape)
    print("✅ Feature-kolonner:", list(features.columns))
    print("NaN:", features.isna().sum().sum())
    assert not features.isna().any().any(), "Der er stadig NaN i datasættet!"
    assert "ema_21" in features.columns, "EMA21 mangler!"
    assert "ema_200" in features.columns, "EMA200 mangler!"
    assert "rsi_14" in features.columns, "RSI14 mangler!"
    assert "rsi_28" in features.columns, "RSI28 mangler!"
    assert "target" in features.columns, "Target mangler!"
    print("✅ Test bestået – alle hovedfeatures findes og ingen NaN!")

if __name__ == "__main__":
    test_pipeline()
    from config.config import COINS, TIMEFRAMES
    for symbol in COINS:
        for tf in TIMEFRAMES:
            raw_path = f"outputs/data/{symbol.lower()}_{tf}_raw.csv"
            if not os.path.exists(raw_path):
                print(f"❌ Data ikke fundet: {raw_path}")
                continue
            raw_df = pd.read_csv(raw_path)
            features = generate_features(raw_df)
            save_features(features, symbol=symbol, timeframe=tf, version="v1.3")
