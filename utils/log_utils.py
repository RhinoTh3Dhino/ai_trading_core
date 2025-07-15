# utils/log_utils.py

import os
import datetime
import platform
import socket

try:
    import torch
except ImportError:
    torch = None

def log_to_file(line, prefix="[INFO] ", log_path="logs/bot.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(prefix + line)

def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return "unknown"

def get_env_info():
    try:
        user = os.getlogin()
    except Exception:
        user = "n/a"
    hostname = socket.gethostname()
    return user, hostname

def log_device_status(
    context="pipeline",
    extra=None,
    botstatus_path="BotStatus.md",
    log_path="logs/bot.log",
    telegram_func=None,
    print_console=True
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch_version = getattr(torch, "__version__", "n/a") if torch else "n/a"
    python_version = platform.python_version()
    user, hostname = get_env_info()
    git_commit = get_git_commit()

    # --- Device detection ---
    if torch:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_mem_alloc = torch.cuda.memory_allocated() // (1024**2)
            cuda_mem_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        else:
            device_name = "CPU"
            cuda_mem_alloc = 0
            cuda_mem_total = 0
    else:
        device = None
        device_str = "cpu"
        device_name = "CPU"
        cuda_mem_alloc = 0
        cuda_mem_total = 0

    status_line = (
        f"{now} | PyTorch {torch_version} | Python {python_version} | "
        f"Device: {device_str.upper()} ({device_name})"
    )
    status_line += f" | Host: {hostname} | User: {user} | Git: {git_commit}"

    if context:
        status_line += f" | Context: {context}"
    if extra is not None:
        if isinstance(extra, dict):
            extra_str = " | ".join([f"{k}: {v}" for k, v in extra.items()])
            status_line += f" | {extra_str}"
        elif isinstance(extra, str):
            status_line += f" | {extra}"

    status_line += "\n"

    # --- Logging til fil, konsol, BotStatus.md og Telegram ---
    if print_console:
        print(f"[BotStatus.md] {status_line.strip()}")
    with open(botstatus_path, "a", encoding="utf-8") as f:
        f.write(status_line)
    log_to_file(status_line, prefix="", log_path=log_path)
    if telegram_func is not None:
        try:
            telegram_func("🤖 " + status_line.strip())
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke sende til Telegram: {e}")

    return {
        "device_str": device_str,
        "device_name": device_name,
        "cuda_mem_alloc": cuda_mem_alloc,
        "cuda_mem_total": cuda_mem_total,
        "torch_version": torch_version,
        "python_version": python_version,
        "user": user,
        "hostname": hostname,
        "git_commit": git_commit,
        "context": context,
        "status_line": status_line.strip(),
    }
