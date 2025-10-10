# config/env_loader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None


def _to_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _to_float(v: str | None, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


@dataclass(frozen=True)
class TelegramCfg:
    token: str
    chat_id: str
    verbosity: str
    min_gap_s: float


@dataclass(frozen=True)
class AlertCfg:
    allow_alerts: bool
    dd_pct: float
    winrate_min: float
    profit_pct: float
    cooldown_s: float


@dataclass(frozen=True)
class EngineCfg:
    log_dir: Path
    alloc_pct: float
    commission_bp: float
    slippage_bp: float
    daily_loss_limit_pct: float
    telegram: TelegramCfg
    alerts: AlertCfg


def load_config(env_path: str | None = None) -> EngineCfg:
    if load_dotenv and (env_path or Path(".env").exists()):
        load_dotenv(dotenv_path=env_path)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))

    telegram = TelegramCfg(
        token=os.getenv("TELEGRAM_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        verbosity=os.getenv("TELEGRAM_VERBOSITY", "trade").lower(),
        min_gap_s=_to_float(os.getenv("TELEGRAM_MIN_SECONDS_BETWEEN_MSG"), 10.0),
    )

    alerts = AlertCfg(
        allow_alerts=_to_bool(os.getenv("ALLOW_ALERTS"), True),
        dd_pct=_to_float(os.getenv("ALERT_DD_PCT"), 10.0),
        winrate_min=_to_float(os.getenv("ALERT_WINRATE_MIN"), 45.0),
        profit_pct=_to_float(os.getenv("ALERT_PROFIT_PCT"), 20.0),
        cooldown_s=_to_float(os.getenv("ALERT_COOLDOWN_SEC"), 1800.0),
    )

    return EngineCfg(
        log_dir=log_dir,
        alloc_pct=_to_float(os.getenv("ENGINE_ALLOC_PCT"), 0.10),
        commission_bp=_to_float(os.getenv("ENGINE_COMMISSION_BP"), 2.0),
        slippage_bp=_to_float(os.getenv("ENGINE_SLIPPAGE_BP"), 1.0),
        daily_loss_limit_pct=_to_float(os.getenv("ENGINE_DAILY_LOSS_LIMIT_PCT"), 5.0),
        telegram=telegram,
        alerts=alerts,
    )
