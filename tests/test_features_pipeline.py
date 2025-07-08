import os
import pandas as pd
import pytest
from datetime import datetime
from features.features_pipeline import generate_features, save_features, load_features

RAW_DATA_PATH = "data/test_data/BTCUSDT_1h_test.csv"
SYMBOL = "BTC"
TIMEFRAME = "1h"
VERSION = "test"

def make_version_with_timestamp(version):
    """Sørg for at version får timestamp-suffix, så load_features virker."""
    ts = datetime.now().strftime("%Y%m%d")
    return f"{version}_{ts}"

def ensure_dir_exists(path):
    """Sikrer at den relevante output-mappe altid findes."""
    os.makedirs(path, exist_ok=True)

def test_generate_features_pipeline():
    # 0. Sikrer output-mappe
    ensure_dir_exists("outputs/feature_data")

    # 1. Læs rådata med semikolon-separator
    assert os.path.exists(RAW_DATA_PATH), f"Rådatafil mangler: {RAW_DATA_PATH}"
    raw_df = pd.read_csv(RAW_DATA_PATH, sep=";")
    assert len(raw_df) > 0, "Rådata er tom"

    # 2. Omdøb "datetime" til "timestamp" hvis nødvendigt
    if "datetime" in raw_df.columns:
        raw_df.rename(columns={"datetime": "timestamp"}, inplace=True)

    # 3. Konverter alle tal-kolonner til float (retter komma til punktum først)
    for col in ["open", "high", "low", "close", "volume"]:
        raw_df[col] = raw_df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

    # 4. Tjek at de nødvendige kolonner findes
    min_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in min_cols:
        assert col in raw_df.columns, f"Kolonne mangler: {col}"

    # 5. Generér features
    features_df = generate_features(raw_df)
    assert not features_df.isnull().values.any(), "Features indeholder NaN!"

    # 6. Tjek at alle centrale kolonner er med
    expected_cols = [
        "rsi_14", "rsi_28", "macd", "macd_signal", "ema_9", "ema_21", "ema_50",
        "atr_14", "vwap", "bb_upper", "bb_lower", "return", "pv_ratio",
        "volume_spike", "regime"
    ]
    missing_features = [col for col in expected_cols if col not in features_df.columns]
    if missing_features:
        print(f"[ADVARSEL] Mangler følgende features: {missing_features}")
    for col in expected_cols:
        assert col in features_df.columns, f"Feature mangler: {col}"

    # 7. Tjek target-kolonne hvis du bruger supervised ML
    if "target" in features_df.columns:
        assert not features_df["target"].isnull().any(), "Target indeholder NaN!"
    else:
        print("[INFO] Ingen target-kolonne fundet (ikke nødvendigt for test hvis pipeline kun genererer features)")

    # 8. Gem features versioneret med timestamp
    version_ts = make_version_with_timestamp(VERSION)
    path = save_features(features_df, symbol=SYMBOL, timeframe=TIMEFRAME, version=version_ts)
    assert os.path.exists(path), f"Featurefil blev ikke gemt: {path}"

    # 9. Genindlæs og tjek igen (nu matcher timestamped version_prefix)
    loaded_df = load_features(SYMBOL, TIMEFRAME, version_prefix=version_ts)
    assert len(loaded_df) > 0, "Indlæst featurefil er tom"
    assert not loaded_df.isnull().values.any(), "Indlæste features indeholder NaN!"

    print("✅ Alle feature-tests bestået!")

if __name__ == "__main__":
    test_generate_features_pipeline()
