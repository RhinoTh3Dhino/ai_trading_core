## tests/persistence/test_artifacts.py


import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.artifacts import rotate_dir, symlink_latest, write_json


def test_write_and_latest(tmp_path):
    out = tmp_path / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    p = write_json({"ok": True}, str(out), "metric_x", "v1")
    latest = out / "metric_x_latest.json"
    symlink_latest(str(p), str(latest))
    assert latest.exists()
    got = json.loads(latest.read_text(encoding="utf-8"))
    assert got["ok"] is True


def test_rotation(tmp_path):
    d = tmp_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (d / f"m_{i}.keras").write_bytes(b"x")
    rotate_dir(str(d), keep=2, pattern=r".*\.keras$")
    files = sorted([p.name for p in d.iterdir()])
    assert len(files) == 2
