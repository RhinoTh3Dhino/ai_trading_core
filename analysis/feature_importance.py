# analysis/feature_importance.py
"""
Feature Importance-analyse for LightGBM/ML trading modeller.
Kør: python analysis/feature_importance.py --input data/BTCUSDT_1h_features.csv --target target_regime_adapt --features close,rsi_14,ema_9 --balance undersample

Argumenter:
--input         CSV-fil med features (påkrævet)
--target        Target-kolonne (fx 'target_regime_adapt')
--features      Kommasepareret feature-liste (fx 'close,rsi_14,ema_9')
--balance       (undersample|oversample) balancering af target (valgfri)
--top_n         Hvor mange vigtigste features der skal vises/gemmes (default: 20)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from utils.project_path import PROJECT_ROOT


def balance_df(df, target, method="undersample", random_state=42, verbose=True):
    counts = df[target].value_counts()
    classes = counts.index.tolist()
    min_class = counts.min()
    max_class = counts.max()
    if verbose:
        print(f"Før balancering: {dict(counts)}")
    dfs = []
    if method == "undersample":
        n = min_class
        for c in classes:
            dfs.append(df[df[target] == c].sample(n=n, random_state=random_state))
        balanced = pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
        if verbose:
            print(f"Undersamplet alle klasser til: {n}")
    elif method == "oversample":
        n = max_class
        for c in classes:
            dfs.append(df[df[target] == c].sample(n=n, replace=True, random_state=random_state))
        balanced = pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
        if verbose:
            print(f"Oversamplet alle klasser til: {n}")
    else:
        raise ValueError(f"Ukendt balanceringsmetode: {method}")
    after_counts = balanced[target].value_counts()
    print(f"Efter balancering: {dict(after_counts)}")
    return balanced


def main():
    parser = argparse.ArgumentParser(
        description="Feature importance-analyse for LightGBM trading-modeller."
    )
    parser.add_argument("--input", type=str, required=True, help="Sti til feature-CSV.")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target-kolonne (fx 'target_regime_adapt').",
    )
    parser.add_argument("--features", type=str, required=True, help="Kommasepareret feature-liste.")
    parser.add_argument(
        "--balance",
        type=str,
        default=None,
        choices=[None, "undersample", "oversample"],
        help="Balancér targets (valgfri).",
    )
    parser.add_argument(
        "--top_n", type=int, default=20, help="Vis kun top N features (default: 20)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()
    features = [col.strip() for col in args.features.split(",")]
    missing = [col for col in features + [args.target] if col not in df.columns]
    if missing:
        print(f"[FEJL] Mangler kolonner i data: {missing}")
        return

    # Balancering hvis valgt
    if args.balance:
        print(f"[INFO] Balancerer target med metode: {args.balance}")
        df = balance_df(df, args.target, method=args.balance)

    X = df[features]
    y = df[args.target].astype(int)

    print(f"[INFO] Træner LightGBM model på {len(df)} rækker, {len(features)} features...")
    model = LGBMClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        by="importance", ascending=False
    )

    print("\n=== TOP FEATURE IMPORTANCE ===")
    print(feat_imp.head(args.top_n))

    # Gem til CSV
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "feature_importance.csv")
    feat_imp.to_csv(out_csv, index=False)
    print(f"[OK] Feature importance gemt i: {out_csv}")

    # Plot
    plt.figure(figsize=(10, 6))
    top_imp = feat_imp.head(args.top_n)
    plt.barh(top_imp["feature"][::-1], top_imp["importance"][::-1])
    plt.xlabel("Importance (split gain)")
    plt.title("Top feature importance (LightGBM)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
