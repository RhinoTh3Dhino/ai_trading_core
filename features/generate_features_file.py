import sys
import os
import pandas as pd

# Sikrer at projekt-roden er på sys.path uanset hvor du kører fra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.features_pipeline import generate_features, save_features

# Læs din rå datafil (juster path/sep hvis nødvendigt)
RAW_DATA_PATH = os.path.join("data", "BTCUSDT_1h.csv")
raw_df = pd.read_csv(RAW_DATA_PATH, sep=";")

# Omdøb "datetime" til "timestamp" hvis nødvendigt
if "timestamp" not in raw_df.columns and "datetime" in raw_df.columns:
    raw_df.rename(columns={"datetime": "timestamp"}, inplace=True)

# Konverter kerne-kolonner til numerisk (fjerner tekstproblemer)
for col in ["open", "high", "low", "close", "volume"]:
    if col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# Tjek at timestamp nu findes – ellers vis kolonnerne!
if "timestamp" not in raw_df.columns:
    print("❌ FEJL: Ingen 'timestamp' kolonne! Kolonner i rådata:", list(raw_df.columns))
    raise SystemExit(1)

# Kør din feature-pipeline
features = generate_features(raw_df)

# Gem feature-matrix med korrekt navn/version
save_features(features, symbol="BTC", timeframe="1h", version="v_test")
