# tests/persistence/test_metadata_and_read_guard.py
from __future__ import annotations

import pandas as pd
import pytest

from utils.artifacts import enforce_read_guard, partition_path, write_parquet


def test_read_guard_accepts_matching_features_version(tmp_layout):
    day = partition_path("BTCUSDT", "1m", "2025-01-02")
    df = pd.DataFrame({"ts": pd.to_datetime([1], unit="ms", utc=True), "open": [1.0]})
    p = day / "part-0001.parquet"
    write_parquet(
        df,
        p,
        meta={
            "schema_version": tmp_layout["SCHEMA_VERSION"],
            "features_version": tmp_layout["FEATURES_VERSION"],
            "generator": "test",
        },
    )
    # Skal ikke raise
    enforce_read_guard([p], tmp_layout["FEATURES_VERSION"])


def test_read_guard_rejects_mismatch(tmp_layout):
    day = partition_path("BTCUSDT", "1m", "2025-01-02")
    df = pd.DataFrame({"ts": pd.to_datetime([1], unit="ms", utc=True), "open": [1.0]})
    p = day / "part-0001.parquet"
    write_parquet(
        df,
        p,
        meta={"schema_version": "1.0.0", "features_version": "OLD", "generator": "test"},
    )
    with pytest.raises(ValueError):
        enforce_read_guard([p], expected_features_version=tmp_layout["FEATURES_VERSION"])
