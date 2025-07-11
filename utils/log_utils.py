# utils/log_utils.py

import os
import datetime
import platform

try:
    import torch
except ImportError:
    torch = None  # Tillader brug uden torch installeret (fx ML-only mode)

def log_to_file(line, prefix="[INFO] ", log_path="logs/bot.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(prefix + line)

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

    # --- Device detection robust ---
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
    # --- Context og ekstra info ---
    if context:
        status_line += f" | Context: {context}"
    if extra is not None:
        if isinstance(extra, dict):
            extra_str = " | ".join([f"{k}: {v}" for k, v in extra.items()])
            status_line += f" | {extra_str}"
        elif isinstance(extra, str):
            status_line += f" | {extra}"

    status_line += "\n"

    # --- Logging til fil, konsol og Telegram ---
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

    # --- Returner device info dictionary, så run_all.py/engine.py aldrig fejler ---
    return {
        "device_str": device_str,
        "device_name": device_name,
        "cuda_mem_alloc": cuda_mem_alloc,
        "cuda_mem_total": cuda_mem_total,
        "torch_version": torch_version,
        "python_version": python_version,
        "context": context,
        "status_line": status_line.strip(),
    }
