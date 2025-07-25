# analysis/feature_distribution.py
"""
Feature-distributionsanalyse for trading datasets.
Giver hurtigt overblik over outliers, bias og datastruktur – før modeltræning!

Brug:
python analysis/feature_distribution.py --input data/BTCUSDT_1h_features.csv --features close,rsi_14,ema_9

Argumenter:
--input     CSV-fil med feature-datasæt (påkrævet)
--features  Kommasepareret liste af features (default: alle numeriske)
--bins      Antal bins til histogrammer (default: 50)
--out_dir   Output-mappe til gemte plots (default: outputs/feature_dist)
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.project_path import PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser(description="Feature-distributionsanalyse for trading/ML datasets.")
    parser.add_argument("--input", type=str, required=True, help="Sti til feature-CSV.")
    parser.add_argument("--features", type=str, default=None, help="Kommasepareret liste af features (default: alle numeriske)")
    parser.add_argument("--bins", type=int, default=50, help="Antal bins i histogrammer (default: 50)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output-mappe for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()
    if args.features:
        features = [col.strip() for col in args.features.split(",")]
    else:
        # Default: alle numeriske kolonner (ekskl. timestamp og targets)
        features = [c for c in df.select_dtypes(include='number').columns if not c.startswith("target") and c != "timestamp"]

    if len(features) == 0:
        print("[FEJL] Ingen features fundet.")
        return

    # Output-mappe (default: outputs/feature_dist)
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "outputs", "feature_dist")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Plotter og gemmer distributioner for {len(features)} features:")
    for feature in features:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[feature].dropna(), bins=args.bins, kde=True, color='dodgerblue', edgecolor='black')
        plt.title(f"Distribution – {feature}")
        plt.xlabel(feature)
        plt.ylabel("Antal")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Gem plot
        fname = os.path.join(out_dir, f"{feature}_distribution.png")
        plt.savefig(fname)
        plt.close()
        print(f"  - [OK] Gemte: {fname}")

    print(f"\n[OK] Alle plots gemt i: {out_dir}")

    # Ekstra: samlet pairplot for features (kan tage lang tid hvis mange features)
    if len(features) <= 7:
        print("[INFO] Plotter pairplot (kan tage tid)...")
        sns.pairplot(df[features].dropna())
        pairplot_path = os.path.join(out_dir, "pairplot.png")
        plt.savefig(pairplot_path)
        plt.close()
        print(f"[OK] Pairplot gemt i: {pairplot_path}")
    else:
        print("[INFO] Skipper pairplot (for mange features).")

if __name__ == "__main__":
    main()
