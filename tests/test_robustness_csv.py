# -*- coding: utf-8 -*-
import csv
import pytest

from bot.utils.csv_utils import safe_read_csv
from bot.utils.errors import EmptyCSVError, MissingColumnsError


def test_empty_csv_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(EmptyCSVError):
        safe_read_csv(str(p))


def test_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b"])
        w.writerow([1, 2])

    with pytest.raises(MissingColumnsError):
        safe_read_csv(str(p), required_columns=["a", "b", "c"])


def test_ok_csv_returns_rows(tmp_path):
    p = tmp_path / "ok.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "c"])
        w.writerow([1, 2, 3])
        w.writerow([4, 5, 6])

    rows = safe_read_csv(str(p), required_columns=["a", "b"])
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"a", "b", "c"}
