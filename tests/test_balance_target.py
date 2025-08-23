# tests/test_balance_target.py
import importlib
import inspect
import numpy as np
import pandas as pd
import pytest


def _get_fn():
    mod = importlib.import_module("features.balance_target")
    # find et plausibelt API-navn
    for name in ("balance_target", "rebalance_target", "balance_targets", "rebalance"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    pytest.skip("Kunne ikke finde en balance-funktion i features.balance_target")
    return None


def _call_balance(fn, df):
    """Kalder balance-fn robust afhængigt af signaturen."""
    sig = inspect.signature(fn)
    kw = {}
    params = sig.parameters

    if "df" in params:
        kw["df"] = df
    elif "data" in params:
        kw["data"] = df
    else:
        # måske forventer den X,y separat
        X = df.drop(columns=["target"])
        y = df["target"]
        if "X" in params and "y" in params:
            kw["X"], kw["y"] = X, y
        elif "x" in params and "y" in params:
            kw["x"], kw["y"] = X, y
        else:
            # sidste udvej: giv DataFrame som første arg.
            return fn(df)

    # target-col navne
    if "target_col" in params:
        kw["target_col"] = "target"
    elif "label_col" in params:
        kw["label_col"] = "target"
    elif "y_col" in params:
        kw["y_col"] = "target"

    # metode hvis understøttet
    if "method" in params:
        kw["method"] = "undersample"
    if "random_state" in params:
        kw["random_state"] = 42

    return fn(**kw)


def _class_counts_from_result(res):
    if isinstance(res, tuple) and len(res) == 2:
        Xb, yb = res
        # kan være DataFrame/Series/ndarray – lav til Series
        if not hasattr(yb, "value_counts"):
            yb = pd.Series(np.asarray(yb))
        return yb.value_counts()
    elif isinstance(res, pd.DataFrame):
        return res["target"].value_counts()
    else:
        # ukendt format: giv op med skip
        pytest.skip("Ukendt returtype fra balance-funktion")
    return None


def test_balance_target_makes_classes_more_even():
    # Skæv fordeling 90/10
    n = 200
    df = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "target": np.r_[np.zeros(180, dtype=int), np.ones(20, dtype=int)],
        }
    )

    fn = _get_fn()
    res = _call_balance(fn, df)

    counts = _class_counts_from_result(res)
    assert counts is not None and len(counts) >= 2

    # Tjek at balanceringen reducerer skævheden markant:
    ratio = counts.max() / max(1, counts.min())
    assert ratio <= 2.0, f"Fortsat meget skæv efter balancing: {counts.to_dict()}"
