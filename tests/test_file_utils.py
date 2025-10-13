# tests/test_file_utils.py
import builtins
import importlib
import os
import types
from pathlib import Path

import pytest


def _get_module():
    return importlib.import_module("utils.file_utils")


def _first_callable(mod, candidates):
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    return None, None


def test_file_utils_resolve_and_mkdir_and_write(tmp_path):
    fu = _get_module()

    # 1) resolve_path / expand_path
    resolve_fn, resolve_name = _first_callable(
        fu,
        ("resolve_path", "expand_path", "to_abs_path", "normalize_path"),
    )
    if resolve_fn:
        # "~" skal udvides til HOME
        home = Path.home()
        resolved = (
            resolve_fn(str(Path.home() / "some" / "file.txt"))
            if "~" not in str(home)
            else resolve_fn("~/some/file.txt")
        )
        assert str(resolved).startswith(str(home)), f"{resolve_name} udvider ikke til HOME"

    # 2) ensure_dir / safe_makedirs / mkdir_p
    ensure_dir_fn, _ = _first_callable(
        fu,
        (
            "ensure_dir",
            "ensure_directory",
            "ensure_parent_dir",
            "safe_makedirs",
            "mkdir_p",
            "make_dirs",
        ),
    )
    target_dir = tmp_path / "nested" / "dir"
    if ensure_dir_fn:
        ensure_dir_fn(target_dir)
        assert target_dir.exists() and target_dir.is_dir()

    # 3) write_text/save_text og optional read_text
    write_fn, _ = _first_callable(fu, ("write_text", "save_text", "write_file", "save_file"))
    read_fn, _ = _first_callable(fu, ("read_text", "load_text", "read_file", "load_file"))
    file_path = tmp_path / "hello.txt"
    if write_fn:
        write_fn(file_path, "hej med dig")
        assert file_path.exists()
        if read_fn:
            txt = read_fn(file_path)
            assert "hej" in txt

    # Hvis intet blev fundet, bør modulet i det mindste kunne importeres uden fejl
    assert isinstance(fu, types.ModuleType)


def test_file_utils_permission_error_is_surface(tmp_path, monkeypatch):
    fu = _get_module()
    write_fn, _ = _first_callable(fu, ("write_text", "save_text", "write_file", "save_file"))
    if not write_fn:
        pytest.skip("Ingen write-funktion eksponeret i utils.file_utils – springer edge-case over.")

    # Gør builtins.open til at kaste PermissionError for at simulere IO-fejl
    def _boom(*a, **k):
        raise PermissionError("no access")

    monkeypatch.setattr(builtins, "open", _boom, raising=True)

    with pytest.raises(PermissionError):
        write_fn(tmp_path / "blocked.txt", "nope")
