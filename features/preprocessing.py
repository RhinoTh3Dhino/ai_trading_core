import pandas as pd
import numpy as np


def normalize_zscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normaliserer udvalgte kolonner med Z-score og tilføjer _z kolonner.
    Returnerer en kopi, så original DataFrame ikke ændres direkte.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col + "_z"] = (df[col] - mean) / std
            else:
                df[col + "_z"] = 0
        else:
            print(
                f"[normalize_zscore] Kolonne {col} ikke fundet i DataFrame – springer over."
            )
    return df


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
        z = np.abs((df[features] - df[features].mean()) / df[features].std())
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
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

    return df


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
        raise ValueError(
            f"Følgende feature kolonner mangler i DataFrame: {missing_cols}"
        )

    # Lav target som fremtidig prisændring (pct. ændring)
    df["target"] = df[target_col].pct_change(periods=-target_shift).shift(-target_shift)

    # Alternativt kan du lave binært signal, fx: 1 hvis target > 0, ellers 0
    # df['target'] = (df['target'] > 0).astype(int)

    # Drop NaN som opstår pga. shift
    df = df.dropna(subset=feature_cols + ["target"])

    # Returner kun features + target
    return df[feature_cols + ["target"]].reset_index(drop=True)
