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
    """Log en linje til fil, sikrer at mappen eksisterer."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(prefix + line)
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke skrive til log-fil: {e}")


def get_git_commit():
    """Hent aktivt git-commit hash, eller returner 'unknown'."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_env_info():
    """Hent brugernavn og hostname (fail-safe for cloud/server)."""
    try:
        user = os.getlogin()
    except Exception:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "n/a"
    hostname = socket.gethostname()
    return user, hostname


def log_device_status(
    context="pipeline",
    extra=None,
    botstatus_path="BotStatus.md",
    log_path="logs/bot.log",
    telegram_func=None,
    print_console=True,
):
    """Logger system-status til BotStatus.md, logfil og evt. Telegram."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch_version = getattr(torch, "__version__", "n/a") if torch else "n/a"
    python_version = platform.python_version()
    user, hostname = get_env_info()
    git_commit = get_git_commit()

    # Device/memory info
    if torch:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_mem_alloc = torch.cuda.memory_allocated() // (1024**2)
            cuda_mem_total = torch.cuda.get_device_properties(0).total_memory // (
                1024**2
            )
            mem_str = f"{cuda_mem_alloc}/{cuda_mem_total}MB"
        else:
            device_name = "CPU"
            cuda_mem_alloc = 0
            cuda_mem_total = 0
            mem_str = "N/A"
    else:
        device = None
        device_str = "cpu"
        device_name = "CPU"
        cuda_mem_alloc = 0
        cuda_mem_total = 0
        mem_str = "N/A"

    status_line = (
        f"{now} | PyTorch {torch_version} | Python {python_version} | "
        f"Device: {device_str.upper()} ({device_name}) | "
        f"VRAM: {mem_str} | "
        f"Host: {hostname} | User: {user} | Git: {git_commit}"
    )

    if context:
        status_line += f" | Context: {context}"
    if extra is not None:
        if isinstance(extra, dict):
            extra_str = " | ".join([f"{k}: {v}" for k, v in extra.items()])
            status_line += f" | {extra_str}"
        elif isinstance(extra, str):
            status_line += f" | {extra}"

    status_line += "\n"

    # Output til konsol, BotStatus.md og logfil (uden emoji i konsol)
    if print_console:
        # Emojis undgås
        print(f"[BotStatus.md] {status_line.strip()}")
    try:
        with open(botstatus_path, "a", encoding="utf-8") as f:
            f.write(status_line)
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke skrive til BotStatus.md: {e}")
    log_to_file(status_line, prefix="", log_path=log_path)

    # Telegram-integration (valgfri)
    if telegram_func is not None:
        try:
            # Her må emojis gerne bruges, da Telegram understøtter det
            telegram_func("Bot-status:\n" + status_line.strip())
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke sende til Telegram: {e}")

    # Returnér info til evt. brug/test
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


# Eksempel/test
if __name__ == "__main__":
    log_device_status(context="unittest", print_console=True)
