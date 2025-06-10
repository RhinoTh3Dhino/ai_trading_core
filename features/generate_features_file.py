import sys
import os
import pandas as pd
import numpy as np

# Sikrer at projekt-roden er på sys.path uanset hvor du kører fra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.features_pipeline import generate_features, save_features

# Læs din rå datafil (juster path/sep hvis nødvendigt)
RAW_DATA_PATH = os.path.join("data", "BTCUSDT_1h.csv")
raw_df = pd.read_csv(RAW_DATA_PATH, sep=";")

print("✅ Indlæst rådata med rækker:", len(raw_df))
print("Kolonner i rådata:", list(raw_df.columns))
print("Eksempel på rådata (top 3 rækker):\n", raw_df.head(3))

# Omdøb "datetime" til "timestamp" hvis nødvendigt
if "timestamp" not in raw_df.columns and "datetime" in raw_df.columns:
    raw_df.rename(columns={"datetime": "timestamp"}, inplace=True)
    print("ℹ️ Omdøbt 'datetime' til 'timestamp'")

# KOMMA → PUNKTUM FIX: Gør alle tal "float-compatible"
for col in ["open", "high", "low", "close", "volume"]:
    if col in raw_df.columns:
        raw_df[col] = (
            raw_df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace("nan", "")
        )
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# Debug: Print hvor mange NaN der nu er i hver kolonne
print("🔎 NaN per kolonne i rådata (efter komma-fix):")
print(raw_df.isna().sum())

# Tjek at timestamp nu findes – ellers vis kolonnerne!
if "timestamp" not in raw_df.columns:
    print("❌ FEJL: Ingen 'timestamp' kolonne! Kolonner i rådata:", list(raw_df.columns))
    raise SystemExit(1)

# --- Debug før pipeline ---
print("Rækker før generate_features():", len(raw_df))

# Kør din feature-pipeline
features = generate_features(raw_df)

# --- NYT: Tilføj dummy-target hvis nødvendigt ---
if "target" not in features.columns:
    features["target"] = np.random.choice([1, 0, -1], size=len(features))
    print("⚠️ Tilføjede dummy 'target' kolonne til features (kun til test)")

# --- Debug efter pipeline ---
print("Rækker efter generate_features():", len(features))
if len(features) == 0:
    print("❌ FEJL: Ingen rækker efter feature-pipeline! Tjek input og rolling windows.")
    print("Eksempel på input-data til pipeline:\n", raw_df.head())
else:
    save_features(features, symbol="BTC", timeframe="1h", version="v_test")
    print("✅ Features gemt – pipeline færdig! Filen ligger i outputs/feature_data/")
