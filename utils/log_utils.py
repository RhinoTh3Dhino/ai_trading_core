# utils/log_utils.py

import os
import datetime
import platform
import torch

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
    torch_version = getattr(torch, "__version__", "n/a")
    python_version = platform.python_version()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_status = "GPU" if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_mem_alloc = torch.cuda.memory_allocated() // (1024**2)
        cuda_mem_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        status_line = (
            f"{now} | PyTorch {torch_version} | Python {python_version} | "
            f"Device: GPU ({device_name}) | CUDA alloc: {cuda_mem_alloc} MB / {cuda_mem_total} MB | "
            f"Context: {context}"
        )
    else:
        status_line = (
            f"{now} | PyTorch {torch_version} | Python {python_version} | "
            f"Device: CPU | Context: {context}"
        )

    # Tilføj ekstra info hvis ønsket (fx strategi, datafil, hyperparametre)
    if extra is not None:
        if isinstance(extra, dict):
            extra_str = " | ".join([f"{k}: {v}" for k, v in extra.items()])
            status_line += f" | {extra_str}"
        elif isinstance(extra, str):
            status_line += f" | {extra}"
    
    status_line += "\n"
    
    if print_console:
        print(f"[BotStatus.md] {status_line.strip()}")
    # Skriv til BotStatus.md
    with open(botstatus_path, "a", encoding="utf-8") as f:
        f.write(status_line)
    # Skriv til log-fil
    log_to_file(status_line, prefix="", log_path=log_path)
    # Send til Telegram hvis funktion gives med
    if telegram_func is not None:
        try:
            telegram_func("🤖 " + status_line.strip())
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke sende til Telegram: {e}")
