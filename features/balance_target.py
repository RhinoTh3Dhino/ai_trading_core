# features/balance_target.py
"""
Balancerer target-klasser i et feature-datasæt via oversampling eller undersampling.
Kan bruges på ALLE targets (fx "target", "target_regime_adapt", "target_tp1.0_sl1.0").

Programmatisk brug:
-------------------
from features.balance_target import balance_target, balance_classes, balance_df

# 1) Kun y (returnerer balanceret y)
y_bal = balance_target(y=y, method="oversample", random_state=0)

# 2) X og y (returnerer (X_bal, y_bal))
X_bal, y_bal = balance_target(X=df.drop(columns=["target"]), y=df["target"], method="undersample")

# 3) Direkte på en DataFrame med target-kolonnen (returnerer balanceret df)
df_bal = balance_target(df, target="target", method="oversample")

CLI brug:
---------
python -m features.balance_target --input data/features.csv --target target \
       --method oversample --output data/features_balanced.csv
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------- Hjælpefunktioner ----------


def _as_series(
    y: Union[pd.Series, np.ndarray, list], name: str = "target"
) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y), name=name)


def _upsample_indices(
    y: pd.Series, *, ratio: float = 1.0, random_state: Optional[int] = 0
) -> np.ndarray:
    """
    Returnerer indeks-array, hvor mindretalsklasser oversamples op til majority * ratio.
    ratio=1.0 -> op til majority-størrelse. ratio>1.0 -> oversamples yderligere.
    """
    rng = np.random.default_rng(random_state)
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return np.array([], dtype=int)

    max_n = int(counts.max() * max(1.0, ratio))
    idxs = []
    for val, n in counts.items():
        cls_idx = y.index[y == val].to_numpy()
        if 0 < n < max_n:
            add = rng.choice(cls_idx, size=max_n - n, replace=True)
            cls_idx = np.concatenate([cls_idx, add])
        idxs.append(cls_idx)
    return np.concatenate(idxs)


def _undersample_indices(
    y: pd.Series, *, ratio: float = 1.0, random_state: Optional[int] = 0
) -> np.ndarray:
    """
    Returnerer indeks-array, hvor alle klasser undersamples ned til minority * ratio.
    ratio=1.0 -> ned til minority-størrelse. ratio<1.0 -> endnu færre (typisk ikke ønsket).
    """
    rng = np.random.default_rng(random_state)
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return np.array([], dtype=int)

    min_n = int(counts.min() * max(0.0, ratio))
    idxs = []
    for val, n in counts.items():
        cls_idx = y.index[y == val].to_numpy()
        if n > min_n and min_n > 0:
            take = rng.choice(cls_idx, size=min_n, replace=False)
            cls_idx = take
        idxs.append(cls_idx)
    return np.concatenate(idxs)


# ---------- Offentlige API-funktioner ----------


def balance_classes(
    X: Optional[pd.DataFrame] = None,
    y: Optional[Union[pd.Series, np.ndarray, list]] = None,
    *,
    target: Optional[str] = None,
    method: str = "oversample",  # "oversample" / "undersample"
    ratio: float = 1.0,
    random_state: Optional[int] = 0,
) -> Union[pd.Series, Tuple[pd.DataFrame, pd.Series]]:
    """
    Simpel, afhængighedsfri class-balancering.

    - Giver du kun y: returnerer en balanceret Series.
    - Giver du X og y: returnerer (X_bal, y_bal), samme rækkefølge/align.
    - Giver du X og target='kolonnenavn' (men ikke y): bruger X[target] som y.

    method: "oversample" eller "undersample"
    ratio : float (typisk 1.0). Ved oversample oversamples op til majority*ratio,
            ved undersample ned til minority*ratio.
    """
    # Lille hjælp: hvis ingen y/target, men X har "target"-kolonnen, brug den.
    if (
        y is None
        and target is None
        and isinstance(X, pd.DataFrame)
        and "target" in X.columns
    ):
        target = "target"

    if y is None:
        if X is None or target is None:
            raise ValueError("Angiv y eller (X og target=kolonnenavn).")
        y = X[target]

    y_s = _as_series(y)
    if y_s.empty:
        # Tomt input -> returner tomt output i samme form
        return (X.iloc[0:0], y_s.iloc[0:0]) if X is not None else y_s

    method_norm = method.lower().strip()
    if method_norm not in {"oversample", "undersample"}:
        raise ValueError(
            f"Ukendt metode '{method}'. Brug 'oversample' eller 'undersample'."
        )

    if method_norm == "oversample":
        idx = _upsample_indices(y_s, ratio=ratio, random_state=random_state)
    else:
        idx = _undersample_indices(y_s, ratio=ratio, random_state=random_state)

    idx = pd.Index(idx)
    y_bal = y_s.loc[idx].reset_index(drop=True)
    if X is None:
        return y_bal

    X_bal = X.loc[idx].reset_index(drop=True)
    return X_bal, y_bal


def balance_df(
    df: pd.DataFrame,
    target: str,
    method: str = "undersample",
    random_state: int = 42,
    verbose: bool = True,
    ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Balancer direkte et DataFrame på target-kolonnen.
    Returnerer et nyt DataFrame i balanceret form.

    method: "oversample" eller "undersample" (default er "undersample").
    """
    if target not in df.columns:
        raise KeyError(
            f"Target '{target}' findes ikke i df! Kolonner: {list(df.columns)}"
        )

    if verbose:
        print(f"Før balancering: {dict(df[target].value_counts())}")

    X_bal, _ = balance_classes(
        X=df, target=target, method=method, ratio=ratio, random_state=random_state
    )

    balanced = X_bal

    if verbose:
        after_counts = balanced[target].value_counts()
        print(f"Efter balancering: {dict(after_counts)}")

    return balanced


def balance_target(*args, **kwargs):
    """
    Bekvem wrapper der understøtter tre brugsmønstre:

    - balance_target(df, target="target", ...)     ->  returnerer balanceret DataFrame
    - balance_target(y=..., ...)                   ->  returnerer balanceret Series
    - balance_target(X=..., y=.../target=..., ...) ->  returnerer (X_bal, y_bal)

    Implementering:
    - Hvis 'X' eller 'y' findes i kwargs, videresend til balance_classes.
    - Hvis første positional-arg er et DataFrame, håndter som balance_df.
    - Ellers antag input er y og balancér som Series.
    """
    if "X" in kwargs or "y" in kwargs:
        return balance_classes(*args, **kwargs)

    if args and isinstance(args[0], pd.DataFrame):
        df = args[0]
        method = kwargs.get("method", "undersample")
        target = kwargs.get("target", "target")
        ratio = kwargs.get("ratio", 1.0)
        random_state = kwargs.get("random_state", 42)
        verbose = kwargs.get("verbose", False)
        return balance_df(
            df,
            target=target,
            method=method,
            random_state=random_state,
            ratio=ratio,
            verbose=verbose,
        )

    # Faldbag: antag det er et y-input
    return balance_classes(*args, **kwargs)


# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(
        description="Balancer target-klasser i feature-CSV."
    )
    parser.add_argument("--input", type=str, required=True, help="Sti til input-CSV")
    parser.add_argument(
        "--target", type=str, default="target", help="Target-kolonne (fx 'target')"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="undersample",
        choices=["undersample", "oversample"],
        help="Balanceringsmetode",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help=(
            "Oversample op til majority*ratio (ved oversample) / "
            "undersample ned til minority*ratio (ved undersample)."
        ),
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Sti til output-CSV (balanceret)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random state/seed")
    args = parser.parse_args()

    print(f"[INFO] Indlæser: {args.input}")
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        print(
            f"❌ FEJL: Target '{args.target}' findes ikke i data! Kolonner: {list(df.columns)}"
        )
        return

    balanced = balance_df(
        df,
        target=args.target,
        method=args.method,
        random_state=args.seed,
        ratio=args.ratio,
        verbose=True,
    )

    balanced.to_csv(args.output, index=False)
    print(f"✅ Balanceret data gemt i: {args.output}")


if __name__ == "__main__":
    main()
