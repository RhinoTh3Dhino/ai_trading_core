# analysis/target_distribution.py
"""
Analyserer og visualiserer fordelingen af targets i din feature-fil.
Viser counts, balance, graf, samt eksport til CSV hvis ønsket.

Kør fx:
python run.py analysis/target_distribution.py --input data/BTCUSDT_1h_features.csv --target target
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.project_path import PROJECT_ROOT


def analyze_target_distribution(df, target_col, output_dir=None, show_plot=True):
    # Tæl target-klasser
    value_counts = df[target_col].value_counts(dropna=False).sort_index()
    print(f"\n=== Target-fordeling for: {target_col} ===")
    print(value_counts)
    print("\nProcentvis fordeling:")
    print((value_counts / len(df) * 100).round(2).astype(str) + "%")

    # Simple stats
    n_total = len(df)
    n_1 = value_counts[1] if 1 in value_counts.index else 0
    n_0 = value_counts[0] if 0 in value_counts.index else 0
    n_minus1 = value_counts[-1] if -1 in value_counts.index else 0
    print(f"\nTotal: {n_total} | n_1: {n_1}, n_0: {n_0}, n_-1: {n_minus1}")

    # Gem til CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"target_dist_{target_col}.csv")
        value_counts.to_csv(out_path)
        print(f"✅ Target-fordeling gemt i: {out_path}")

    # Plot (valgfri)
    if show_plot:
        plt.figure(figsize=(6, 4))
        value_counts.plot(
            kind="bar",
            color=[
                "#388e3c" if i == 1 else "#1976d2" if i == 0 else "#b71c1c"
                for i in value_counts.index
            ],
        )
        plt.title(f"Target-fordeling: {target_col}")
        plt.xlabel("Klasse")
        plt.ylabel("Antal")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analysér target-fordeling i feature-fil.")
    parser.add_argument("--input", type=str, required=True, help="Sti til feature-CSV")
    parser.add_argument("--target", type=str, default="target", help="Navn på target-kolonne")
    parser.add_argument("--output_dir", type=str, default=None, help="Mappe til eksport af CSV")
    parser.add_argument("--no_plot", action="store_true", help="Skjul plot")
    args = parser.parse_args()

    print(f"[INFO] Indlæser data: {args.input}")
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        print(f"❌ FEJL: Target '{args.target}' findes ikke i data! Kolonner: {list(df.columns)}")
        return

    analyze_target_distribution(
        df,
        target_col=args.target,
        output_dir=args.output_dir,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
