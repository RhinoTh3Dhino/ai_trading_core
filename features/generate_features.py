# features/generate_features.py

import argparse
import pandas as pd
import numpy as np
import os
import sys
from utils.project_path import PROJECT_ROOT
from features.features_pipeline import generate_features, save_features

def auto_detect_sep(filepath):
    with open(filepath, 'r') as f:
        header = f.readline()
        if ";" in header and not "," in header:
            return ";"
        return ","

def main():
    parser = argparse.ArgumentParser(description="Generér features fra rå OHLCV-data.")
    parser.add_argument("--input", type=str, required=True, help="Sti til rå OHLCV-data (CSV)")
    parser.add_argument("--output", type=str, default=None, help="Sti til output-CSV (fx data/BTCUSDT_1h_features.csv)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (fx BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (fx 1h, 4h)")
    parser.add_argument("--version", type=str, default="v1", help="Feature-version")
    parser.add_argument("--sep", type=str, default=None, help="Separator i input-CSV (auto-detect hvis tom)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ FEJL: Filen '{args.input}' findes ikke.")
        sys.exit(1)

    sep = args.sep or auto_detect_sep(args.input)
    print(f"[INFO] Bruger separator: '{sep}'")

    raw_df = pd.read_csv(args.input, sep=sep)

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

    print("🔎 NaN per kolonne i rådata (efter komma-fix):")
    print(raw_df.isna().sum())

    # Tjek at timestamp nu findes – ellers vis kolonnerne!
    if "timestamp" not in raw_df.columns:
        print("❌ FEJL: Ingen 'timestamp' kolonne! Kolonner i rådata:", list(raw_df.columns))
        sys.exit(1)

    print("Rækker før generate_features():", len(raw_df))

    # Kør din feature-pipeline
    features = generate_features(raw_df)

    # --- NYT: Tilføj dummy-target hvis nødvendigt ---
    if "target" not in features.columns:
        features["target"] = np.random.choice([1, 0], size=len(features))
        print("⚠️ Tilføjede dummy 'target' kolonne til features (kun til test)")

    print("Rækker efter generate_features():", len(features))
    if len(features) == 0:
        print("❌ FEJL: Ingen rækker efter feature-pipeline! Tjek input og rolling windows.")
        print("Eksempel på input-data til pipeline:\n", raw_df.head())
        sys.exit(1)
    else:
        # Gem til valgt output-fil – enten direkte, eller i systemstruktur
        if args.output:
            features.to_csv(args.output, index=False)
            print(f"✅ Features gemt – pipeline færdig! Filen ligger i: {args.output}")
        else:
            path = save_features(features, args.symbol, args.timeframe, args.version)
            print(f"✅ Features gemt: {path}")

if __name__ == "__main__":
    main()
