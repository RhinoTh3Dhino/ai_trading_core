# tests/persistence/test_dedup_and_compact.py
from __future__ import annotations

import pandas as pd

from utils.artifacts import compact_day, next_part_path, partition_path, write_parquet


def test_dedup_and_compact_reduces_files_and_rows(tmp_layout):
    day = partition_path("BTCUSDT", "1m", "2025-01-02")

    # Lav 3 små part-filer med overlap på ts
    def make_df(ts_list):
        return pd.DataFrame(
            {
                "ts": pd.to_datetime(ts_list, unit="ms", utc=True),
                "open": [1.0 + i * 0.01 for i in range(len(ts_list))],
                "high": [1.1] * len(ts_list),
                "low": [0.9] * len(ts_list),
                "close": [1.05] * len(ts_list),
                "volume": [10.0] * len(ts_list),
            }
        )

    f1 = next_part_path(day)
    write_parquet(
        make_df([1_000, 2_000, 3_000]),
        f1,
        meta={
            "schema_version": "1.0.0",
            "features_version": tmp_layout["FEATURES_VERSION"],
            "generator": "t",
        },
    )

    f2 = next_part_path(day)
    write_parquet(
        make_df([2_000, 3_000, 4_000]),
        f2,
        meta={
            "schema_version": "1.0.0",
            "features_version": tmp_layout["FEATURES_VERSION"],
            "generator": "t",
        },
    )

    f3 = next_part_path(day)
    write_parquet(
        make_df([3_000, 4_000, 5_000]),
        f3,
        meta={
            "schema_version": "1.0.0",
            "features_version": tmp_layout["FEATURES_VERSION"],
            "generator": "t",
        },
    )

    stats = compact_day(day, expected_features_version=tmp_layout["FEATURES_VERSION"])
    # Før: 3 filer * 3 rækker = 9; unikke ts = 5
    assert stats["in_files"] == 3
    assert stats["in_rows"] == 9
    assert stats["out_rows"] == 5
    assert stats["dropped_dups"] == 4
    # Antal outputfiler styres af ROTATE_MAX_ROWS=2 → ceil(5/2)=3
    assert stats["out_files"] == 3
