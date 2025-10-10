# utils/log_utils.py
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch  # type: ignore
except ImportError:
    torch = None


def _ensure_parent_dir(path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_to_file(line: str, prefix: str = "", log_path: str = "logs/bot.log") -> None:
    try:
        _ensure_parent_dir(log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(prefix + line + "\n")
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke skrive til log-fil: {e}", file=sys.stderr)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_env_info() -> Tuple[str, str]:
    try:
        user = os.getlogin()
    except Exception:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "n/a"
    return user, socket.gethostname()


@dataclass
class DeviceInfo:
    device_str: str
    device_name: str
    cuda_mem_alloc_mb: int
    cuda_mem_total_mb: int


def _get_device_info() -> DeviceInfo:
    if torch and getattr(torch, "cuda", None):
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                alloc = int(torch.cuda.memory_allocated(0) // (1024**2))
                total = int(
                    torch.cuda.get_device_properties(0).total_memory // (1024**2)
                )
                return DeviceInfo("cuda", name, alloc, total)
        except Exception:
            pass
    return DeviceInfo("cpu", "CPU", 0, 0)


def log_device_status(
    context: str = "pipeline",
    extra: Optional[Dict[str, Any] | str] = None,
    botstatus_path: str = "BotStatus.md",
    log_path: str = "logs/bot.log",
    telegram_func: Optional[Any] = None,
    print_console: bool = True,
) -> Dict[str, Any]:
    now = _iso_now()
    torch_v = getattr(torch, "__version__", "n/a") if torch else "n/a"
    py_v = platform.python_version()
    user, host = get_env_info()
    git = get_git_commit()
    dev = _get_device_info()
    mem = (
        f"{dev.cuda_mem_alloc_mb}/{dev.cuda_mem_total_mb}MB"
        if dev.device_str == "cuda" and dev.cuda_mem_total_mb > 0
        else "N/A"
    )
    line = f"{now} | PyTorch {torch_v} | Python {py_v} | Device: {dev.device_str.upper()} ({dev.device_name}) | VRAM: {mem} | Host: {host} | User: {user} | Git: {git}"
    if context:
        line += f" | Context: {context}"
    if isinstance(extra, dict):
        line += " | " + " | ".join(f"{k}: {v}" for k, v in extra.items())
    elif isinstance(extra, str):
        line += " | " + extra
    if print_console:
        print(f"[BotStatus.md] {line}")
    try:
        _ensure_parent_dir(botstatus_path)
        with open(botstatus_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(
            f"[ADVARSEL] Kunne ikke skrive til {botstatus_path}: {e}", file=sys.stderr
        )
    log_to_file(line, log_path=log_path)
    if telegram_func:
        try:
            telegram_func("Bot-status:\n" + line)
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke sende til Telegram: {e}", file=sys.stderr)
    return {
        "device_str": dev.device_str,
        "device_name": dev.device_name,
        "cuda_mem_alloc": dev.cuda_mem_alloc_mb,
        "cuda_mem_total": dev.cuda_mem_total_mb,
        "torch_version": torch_v,
        "python_version": py_v,
        "user": user,
        "hostname": host,
        "git_commit": git,
        "context": context,
        "status_line": line,
    }


def rotate_text_log(
    log_path: str | os.PathLike, keep_last_lines: int = 100_000
) -> None:
    p = Path(log_path)
    if not p.exists():
        return
    keep = max(1, int(keep_last_lines))
    block = 1024 * 1024
    lines: list[str] = []
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0 and len(lines) <= keep:
            n = min(block, pos)
            pos -= n
            f.seek(pos)
            chunk = f.read(n)
            buf = chunk + buf
            parts = buf.split(b"\n")
            buf = parts[0]
            for part in reversed(parts[1:]):
                try:
                    lines.append(part.decode("utf-8", errors="ignore"))
                except Exception:
                    lines.append(part.decode("latin-1", errors="ignore"))
                if len(lines) >= keep:
                    break
        if buf:
            try:
                lines.append(buf.decode("utf-8", errors="ignore"))
            except Exception:
                lines.append(buf.decode("latin-1", errors="ignore"))
    lines = list(reversed(lines))[-keep:]
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="") as out:
        out.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, p)


def tail_text_log(log_path: str | os.PathLike, n: int = 200) -> str:
    p = Path(log_path)
    if not p.exists():
        return ""
    keep = max(1, int(n))
    res: list[str] = []
    block = 128 * 1024
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0 and len(res) <= keep:
            n = min(block, pos)
            pos -= n
            f.seek(pos)
            chunk = f.read(n)
            buf = chunk + buf
            parts = buf.split(b"\n")
            buf = parts[0]
            for part in reversed(parts[1:]):
                try:
                    res.append(part.decode("utf-8", errors="ignore"))
                except Exception:
                    res.append(part.decode("latin-1", errors="ignore"))
                if len(res) >= keep:
                    break
        if buf and len(res) < keep:
            try:
                res.append(buf.decode("utf-8", errors="ignore"))
            except Exception:
                res.append(buf.decode("latin-1", errors="ignore"))
    return "\n".join(list(reversed(res))[-keep:])


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Log utils (status/rotate/tail)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rotate", type=str, help="Trim tekstlog til sidste N linjer")
    ap.add_argument("--keep", type=int, default=100_000)
    ap.add_argument("--tail", type=str, help="Vis de sidste N linjer")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.status:
        log_device_status(context="cli", print_console=True)
        return
    if a.rotate:
        rotate_text_log(a.rotate, keep_last_lines=a.keep)
        print(f"Trimmet {a.rotate} til sidste {a.keep} linjer")
        return
    if a.tail:
        print(tail_text_log(a.tail, n=a.n))
        return
    ap.print_help()


if __name__ == "__main__":
    _cli()
