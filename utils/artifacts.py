import json
import os
import re
import shutil
from datetime import datetime
from typing import Optional

DATE_FMT = "%Y%m%d"
DT_FMT = "%Y%m%d_%H%M%S"


def _ts(dt: bool = False) -> str:
    return datetime.now().strftime(DT_FMT if dt else DATE_FMT)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stamp_name(prefix: str, version: str, ext: str, with_time: bool = False) -> str:
    t = _ts(dt=with_time)
    return f"{prefix}_{version}_{t}.{ext.lstrip('.')}"


def write_json(obj: dict, out_dir: str, prefix: str, version: str, with_time=False) -> str:
    ensure_dir(out_dir)
    fn = stamp_name(prefix, version, "json", with_time)
    p = os.path.join(out_dir, fn)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


def write_text(
    text: str, out_dir: str, prefix: str, version: str, with_time=False, ext="txt"
) -> str:
    ensure_dir(out_dir)
    fn = stamp_name(prefix, version, ext, with_time)
    p = os.path.join(out_dir, fn)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def symlink_latest(path: str, latest_link: str):
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.abspath(path), latest_link)
    except Exception:
        shutil.copy2(path, latest_link)  # Windows fallback


def rotate_dir(path: str, keep: int = 30, pattern: Optional[str] = None):
    if not os.path.isdir(path):
        return
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if pattern:
        rx = re.compile(pattern)
        files = [f for f in files if rx.search(os.path.basename(f))]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for f in files[keep:]:
        try:
            os.remove(f)
        except Exception:
            pass
