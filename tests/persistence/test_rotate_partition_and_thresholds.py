# tests/persistence/test_rotate_partition_and_thresholds.py
from __future__ import annotations

import pandas as pd

from utils.artifacts import next_part_path, partition_path, rotate_partition


def test_rotate_partition_creates_next_part(tmp_layout, small_df):
    # Første rotation → part-0001
    day = partition_path("ETHUSDT", "1m", "2025-01-02")
    p1 = next_part_path(day)
    out = rotate_partition("ETHUSDT", "1m", small_df.iloc[:2])  # 2 rækker
    assert str(out).endswith("part-0001.parquet")
    assert p1.exists()

    # Anden rotation → part-0002
    p2 = next_part_path(day)
    rotate_partition("ETHUSDT", "1m", small_df.iloc[2:3])
    assert p2.name.endswith("part-0002.parquet")
