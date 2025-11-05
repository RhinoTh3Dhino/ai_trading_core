# tests/data_quality/test_contracts.py
import pandas as pd

from utils.data_contracts import ColumnSpec, DataContract
from utils.data_quality import validate


def test_missing_columns_fails():
    df = pd.DataFrame({"open": [1.0]})
    c = DataContract(
        name="ohlcv",
        required_cols={
            "timestamp": ColumnSpec("datetime"),
            "open": ColumnSpec("float", min_val=0),
        },
        key_cols=("timestamp",),
    )
    rep = validate(df, c)
    assert not rep.ok
    assert "missing_columns" in rep.issues


def test_dup_rate_detected():
    df = pd.DataFrame({"timestamp": [1, 1, 2], "open": [1.0, 1.0, 2.0]})
    c = DataContract(
        name="ohlcv",
        required_cols={
            "timestamp": ColumnSpec("datetime"),
            "open": ColumnSpec("float"),
        },
        key_cols=("timestamp",),
        max_dup_rate=0.0,
    )
    rep = validate(df, c)
    assert "dup_rate" in rep.issues


def test_bounds_and_nulls():
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [1.0, -1.0, None],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 0.3, 0.1],
            "close": [1.1, 2.2, 3.3],
            "volume": [0.0, 1.0, 2.0],
        }
    )
    c = DataContract(
        name="ohlcv",
        required_cols={
            "timestamp": ColumnSpec("datetime"),
            "open": ColumnSpec("float", min_val=0),
            "high": ColumnSpec("float", min_val=0),
            "low": ColumnSpec("float", min_val=0),
            "close": ColumnSpec("float", min_val=0),
            "volume": ColumnSpec("float", min_val=0),
        },
        key_cols=("timestamp",),
        max_null_rate=0.0,
    )
    rep = validate(df, c)
    assert not rep.ok
    assert (
        "bounds_min" in rep.issues or "null_rate" in rep.issues or "global_null_rate" in rep.issues
    )
