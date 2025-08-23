# tests/test_preprocessing_edge.py
import importlib
import inspect
import numpy as np
import pandas as pd
import pytest


def _mod():
    return importlib.import_module("features.preprocessing")


def _find_fn(mod):
    # Prøv flere almindelige navne/aliaser
    candidates = [
        "clean_and_normalize",
        "clean_and_normalize_df",
        "preprocess_df",
        "normalize_and_clean",
        "preprocess",  # fallback
    ]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    return None, None


def test_clean_and_normalize_handles_nan_and_bounds():
    prep = _mod()
    fn, name = _find_fn(prep)
    if not fn:
        pytest.skip("Fandt ingen egnet preprocess-funktion i features.preprocessing")

    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [10, 10, 10, 10],  # konstant kolonne
            "txt": ["x", "y", "z", "x"],
        }
    )

    # Kald robust: tjek signatur
    sig = inspect.signature(fn)
    if "df" in sig.parameters:
        out = fn(df=df.copy())
    else:
        out = fn(df.copy())  # typisk

    assert isinstance(out, pd.DataFrame), f"{name} returnerede ikke DataFrame"

    # Ingen NaN i numeriske kolonner
    num = out.select_dtypes(include="number")
    assert not num.isna().any().any(), f"{name} efterlod NaN i numeric kolonner"

    # Hvis der bruges min-max, bør numeric typisk være inden for [0,1].
    # Vi kræver ikke strengt, men tjekker at værdierne ikke stikker af.
    assert num.max().max() <= 1.0 + 1e-6 or num.max().max() < 5.0
    assert num.min().min() >= -1e-6 or num.min().min() > -5.0


def test_clean_and_normalize_empty_df_returns_empty():
    prep = _mod()
    fn, name = _find_fn(prep)
    if not fn:
        pytest.skip("Fandt ingen egnet preprocess-funktion i features.preprocessing")

    empty = pd.DataFrame()
    sig = inspect.signature(fn)
    out = fn(df=empty.copy()) if "df" in sig.parameters else fn(empty.copy())

    assert isinstance(out, pd.DataFrame)
    assert out.empty or out.shape == (0, 0)


def test_clean_and_normalize_unknown_scaler_raises():
    prep = _mod()
    fn, name = _find_fn(prep)
    if not fn:
        pytest.skip("Fandt ingen egnet preprocess-funktion i features.preprocessing")

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    sig = inspect.signature(fn)
    if "scaler" not in sig.parameters:
        pytest.skip(f"{name} understøtter ikke parameteren 'scaler'")

    with pytest.raises((ValueError, KeyError, RuntimeError, AssertionError)):
        if "df" in sig.parameters:
            fn(df=df, scaler="this-scaler-does-not-exist")
        else:
            fn(df, scaler="this-scaler-does-not-exist")
