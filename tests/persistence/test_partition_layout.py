# tests/persistence/test_partition_layout.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.artifacts import next_part_path, partition_path, read_metadata, write_parquet


def test_partition_layout_and_next_part(tmp_layout):
    # Arrange
    day = partition_path("BTCUSDT", "1m", "2025-01-02")
    p1 = next_part_path(day)
    p2 = next_part_path(day)  # endnu ikke skrevet—skal stadig være part-0001

    # Assert at første path bliver 'part-0001.parquet'
    assert p1.name.endswith("part-0001.parquet")
    assert p1 == p2  # da ingen filer er skrevet endnu

    # Skriv første fil
    df = pd.DataFrame({"ts": pd.to_datetime([1], unit="ms", utc=True), "open": [1.0]})
    write_parquet(
        df,
        p1,
        meta={
            "schema_version": tmp_layout["SCHEMA_VERSION"],
            "features_version": tmp_layout["FEATURES_VERSION"],
            "generator": "test",
        },
    )

    # Næste fil bør blive part-0002
    p3 = next_part_path(day)
    assert p3.name.endswith("part-0002.parquet")

    # Metadata skal være skrevet
    md = read_metadata(p1)
    assert md["schema_version"] == tmp_layout["SCHEMA_VERSION"]
    assert md["features_version"] == tmp_layout["FEATURES_VERSION"]
