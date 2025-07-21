# features/generate_features.py


import argparse
import pandas as pd
import numpy as np



from features.features_pipeline import generate_features, save_features

def main():
    parser = argparse.ArgumentParser(description="Generér features fra rå OHLCV-data.")
    parser.add_argument("--input", type=str, required=True, help="Sti til rå OHLCV-data (CSV)")
    parser.add_argument("--symbol", type=str, default="BTC", help="Symbol (fx BTC, ETH)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (fx 1h, 4h)")
    parser.add_argument("--version", type=str, default="v1.0", help="Feature-version")
    parser.add_argument("--sep", type=str, default=",", help="Separator i input-CSV (default: ',')")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ FEJL: Filen '{args.input}' findes ikke.")
        sys.exit(1)

    raw_df = pd.read_csv(args.input, sep=args.sep)

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
        sys.exit(1)

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
        sys.exit(1)
    else:
        save_features(features, symbol=args.symbol, timeframe=args.timeframe, version=args.version)
        print("✅ Features gemt – pipeline færdig! Filen ligger i outputs/feature_data/")

if __name__ == "__main__":
    main()
