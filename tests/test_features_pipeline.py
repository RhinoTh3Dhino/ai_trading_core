import os
import pandas as pd
import pytest
from features.features_pipeline import generate_features, save_features, load_features

RAW_DATA_PATH = "test_data/BTCUSDT_1h_test.csv"
SYMBOL = "BTC"
TIMEFRAME = "1h"
VERSION = "test"

def test_generate_features_pipeline():
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
    for col in expected_cols:
        assert col in features_df.columns, f"Feature mangler: {col}"

    # 7. Gem features versioneret
    path = save_features(features_df, symbol=SYMBOL, timeframe=TIMEFRAME, version=VERSION)
    assert os.path.exists(path), f"Featurefil blev ikke gemt: {path}"

    # 8. Genindlæs og tjek igen
    loaded_df = load_features(SYMBOL, TIMEFRAME, version_prefix=VERSION)
    assert len(loaded_df) > 0, "Indlæst featurefil er tom"
    assert not loaded_df.isnull().values.any(), "Indlæste features indeholder NaN!"

    print("✅ Alle feature-tests bestået!")

if __name__ == "__main__":
    test_generate_features_pipeline()
