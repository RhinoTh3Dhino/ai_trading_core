# features/preprocessing.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Hjælpefunktioner til skalering
# -----------------------------
def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std and std > 0:
        return (series - mean) / std
    # Konstant kolonne -> 0
    return pd.Series(0.0, index=series.index, dtype=float)


def _minmax(series: pd.Series) -> pd.Series:
    mn = series.min()
    mx = series.max()
    rng = mx - mn
    if rng and rng > 0:
        return (series - mn) / rng
    # Konstant kolonne -> 0
    return pd.Series(0.0, index=series.index, dtype=float)


# ---------------------------------
# 1) Ekisterende funktion (beholdt)
# ---------------------------------
def normalize_zscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normaliserer udvalgte kolonner med Z-score og tilføjer _z kolonner.
    Returnerer en kopi, så original DataFrame ikke ændres direkte.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col + "_z"] = _zscore(df[col])
        else:
            print(f"[normalize_zscore] Kolonne {col} ikke fundet i DataFrame – springer over.")
    return df


# ------------------------------------------------
# 2) Ny generel clean_and_normalize til edge-cases
# ------------------------------------------------
def clean_and_normalize(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    *,
    scaler: str = "zscore",                    # 'zscore' | 'minmax' | 'none'
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    dropna: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Rens og normalisér et DataFrame – robust overfor NaN/inf, tomt input og
    ukendte scalere.

    Steps:
      1) Udskift +/-inf med NaN
      2) (valgfrit) Drop NaN-rækker
      3) (valgfrit) Clip værdier i features til 'bounds' = (lower, upper)
      4) Skalér features ift. valgt 'scaler' (zscore/minmax/none)

    Returns:
      Ny kopi af df med normaliserede værdier (samme kolonnenavne).

    Edge-cases:
      - Tomt df returneres tomt
      - Ukendt scaler -> ValueError
      - Konstant kolonne -> nul-vektor efter skalering
    """
    df = df.copy()

    if df.empty:
        if verbose:
            print("[clean_and_normalize] Tomt DataFrame – returnerer tomt.")
        return df

    # 1) Erstat inf med NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Default: alle numeriske kolonner
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [c for c in features if c in df.columns]

    if not features:
        if verbose:
            print("[clean_and_normalize] Ingen features at normalisere – returnerer kopi.")
        return df

    # 2) Drop NaN hvis ønsket (kun i de features vi arbejder på)
    if dropna:
        before = len(df)
        df = df.dropna(subset=list(features))
        if verbose:
            print(f"[clean_and_normalize] Droppede {before - len(df)} rækker pga. NaN/inf i features.")

    if df.empty:
        return df

    # 3) Clip til bounds
    if bounds is not None:
        lo, hi = bounds
        df.loc[:, features] = df.loc[:, features].clip(lower=lo, upper=hi)

    # 4) Skalering
    scaler_key = scaler.strip().lower()
    if scaler_key not in {"zscore", "minmax", "none"}:
        raise ValueError(f"Ukendt scaler '{scaler}'. Brug 'zscore', 'minmax' eller 'none'.")

    if scaler_key == "zscore":
        for col in features:
            df[col] = _zscore(df[col])
    elif scaler_key == "minmax":
        for col in features:
            df[col] = _minmax(df[col])
    # else: 'none' -> ingen skalering

    return df


# --------------------------------------------
# 3) Din eksisterende clean_dataframe (beholdt)
# --------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    features: list = None,
    outlier_z: float = 3.0,
    dropna: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Rens DataFrame: udskift inf, drop NaN, fjern outliers, (optionelt) normalisér features.
    Returnerer en kopi, så original DataFrame ikke ændres direkte.
    """
    df = df.copy()
    # 1. Udskift inf/-inf med NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Drop NaN hvis ønsket
    if dropna:
        before = len(df)
        df = df.dropna()
        after = len(df)
        print(f"[DataCleaning] Droppede {before - after} rækker med NaN/inf.")

    # 3. Outlier clip (z-score > outlier_z)
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in features if col in df.columns]
    if features:
        # Guard mod div/0 ved std=0: brug np.where for sikkerhed
        means = df[features].mean()
        stds = df[features].std().replace(0, np.nan)
        z = np.abs((df[features] - means) / stds)
        z = z.fillna(0.0)
        mask = (z < outlier_z).all(axis=1)
        before = len(df)
        df = df[mask]
        after = len(df)
        print(f"[DataCleaning] Droppede {before - after} outliers (z > {outlier_z}).")
    else:
        print("[DataCleaning] Ingen numeriske features fundet til outlier check.")

    # 4. Optional: Normaliser features på stedet (uden '_z' suffix)
    if normalize and features:
        for col in features:
            df[col] = _zscore(df[col])

    return df


# ------------------------------------------
# 4) LSTM utilities (beholdt uændret)
# ------------------------------------------
def create_lstm_sequences(df: pd.DataFrame, seq_length: int = 48):
    """
    Omdan DataFrame til LSTM-sekvenser (X, y) til ML-træning.
    Returnerer numpy-arrays klar til ML (X, y).
    """
    X, y = [], []
    data = df.values
    for i in range(len(data) - seq_length - 1):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def prepare_ml_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "close",
    target_shift: int = -1,
) -> pd.DataFrame:
    """
    Forbereder data til ML-træning ved at:
    - Vælge features
    - Lave target som f.eks. næste periodes prisændring eller binært signal
    - Skifte target med target_shift (f.eks. -1 for næste periode)
    - Droppe NaN der opstår pga. shift

    Args:
        df: Input DataFrame med rådata og features.
        feature_cols: Liste over kolonner der skal bruges som features.
        target_col: Kolonnenavn for target, typisk "close".
        target_shift: Hvor mange rækker target skal forskydes (negativ for fremtid).

    Returnerer:
        DataFrame med features og 'target' kolonne klar til ML.
    """
    df = df.copy()
    # Check feature cols findes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Følgende feature kolonner mangler i DataFrame: {missing_cols}")

    # Lav target som fremtidig prisændring (pct. ændring)
    df["target"] = df[target_col].pct_change(periods=-target_shift).shift(-target_shift)

    # Alternativt binært signal:
    # df['target'] = (df['target'] > 0).astype(int)

    # Drop NaN som opstår pga. shift
    df = df.dropna(subset=feature_cols + ["target"])

    # Returner kun features + target
    return df[feature_cols + ["target"]].reset_index(drop=True)
