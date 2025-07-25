# features/balance_target.py
"""
Balancerer target-klasser i et feature-datasæt via oversampling eller undersampling.
Kan bruges på ALLE targets (fx "target", "target_regime_adapt", "target_tp1.0_sl1.0").
Gemmer balanceret fil + rapport.

Eksempel:
python run.py features/balance_target.py --input data/BTCUSDT_1h_features.csv --target target --method oversample --output data/BTCUSDT_1h_features_balanced.csv
"""

import argparse
import pandas as pd
import numpy as np
import os
from utils.project_path import PROJECT_ROOT

def balance_df(df, target, method="undersample", random_state=42, verbose=True):
    counts = df[target].value_counts()
    classes = counts.index.tolist()
    min_class = counts.min()
    max_class = counts.max()

    if verbose:
        print(f"Før balancering: {dict(counts)}")

    # Gør kun på binært target (0/1 eller evt. -1/1)
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
    parser = argparse.ArgumentParser(description="Balancer target-klasser i feature-CSV.")
    parser.add_argument("--input", type=str, required=True, help="Sti til input-CSV")
    parser.add_argument("--target", type=str, default="target", help="Target-kolonne (fx 'target')")
    parser.add_argument("--method", type=str, default="undersample", choices=["undersample", "oversample"], help="Balanceringsmetode")
    parser.add_argument("--output", type=str, required=True, help="Sti til output-CSV (balanceret)")
    args = parser.parse_args()

    print(f"[INFO] Indlæser: {args.input}")
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        print(f"❌ FEJL: Target '{args.target}' findes ikke i data! Kolonner: {list(df.columns)}")
        return

    balanced = balance_df(df, args.target, method=args.method)

    balanced.to_csv(args.output, index=False)
    print(f"✅ Balanceret data gemt i: {args.output}")

if __name__ == "__main__":
    main()
