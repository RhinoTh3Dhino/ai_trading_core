# config/config.py
# --- Central konfiguration for AI trading bot (modulær & SaaS-ready) ---

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

# === Feature-grupper ===
FEATURES: Dict[str, List[str]] = {
    "trend": ["ema_9", "ema_21", "ema_50", "ema_200", "macd"],
    "momentum": ["rsi_14", "rsi_28"],  # evt. "stochastic_14"
    "volatility": ["atr_14", "bb_upper", "bb_lower"],
    "volume": ["vwap"],  # evt. "obv"
    "regime": ["adx_14", "zscore_20", "regime"],
    "extra": [],  # tom liste giver ekstra test-coverage
}

# === Automatisk ALL_FEATURES (unik liste, ingen dublikater/kommenterede) ===
ALL_FEATURES: List[str] = []
for group in FEATURES.values():
    ALL_FEATURES.extend([f for f in group if f and not str(f).strip().startswith("#")])
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))  # Unique, original rækkefølge

# === Coin/strategi-opsætning fra coins_config.py (fallback hvis ikke fundet) ===
try:
    from config.coins_config import COIN_CONFIGS, ENABLED_COINS, ENABLED_TIMEFRAMES
except ImportError:  # pragma: no cover
    COIN_CONFIGS = {}
    ENABLED_COINS = ["BTCUSDT"]
    ENABLED_TIMEFRAMES = ["1h"]

COINS: List[str] = ENABLED_COINS
TIMEFRAMES: List[str] = ENABLED_TIMEFRAMES

# === (Fallback: Default global SL/TP hvis ikke defineret pr. coin) ===
STOP_LOSS: float = 0.02  # 2%
TAKE_PROFIT: float = 0.04  # 4%
RISK_LEVELS: List[float] = [0.01, 0.02, 0.03]

# === Regime-detektion (fx bull/bear thresholds) ===
REGIME_THRESHOLDS = {
    "bull": {"adx_min": 20, "macd_slope": 0.0},
    "bear": {"adx_min": 20, "macd_slope": 0.0},
}

# === Path-opsætning (kan overskrives via env eller CLI) ===
DATA_PATH = os.getenv("DATA_PATH", "data/")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
LOG_PATH = os.getenv("LOG_PATH", "logs/")
OUTPUTS_PATH = os.getenv("OUTPUTS_PATH", "outputs/")  # Ny: persistensrod

# === Telegram/Notification (sæt rigtige værdier via .env i produktion) ===
TELEGRAM = {
    "token": os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID"),
}

# === ML/DL modelparametre ===
ML_PARAMS = {
    "batch_size": 64,
    "epochs": 30,
    "seq_length": 48,
    "learning_rate": 0.001,
}

# === Ensemble/strategi-weights ===
ENSEMBLE_WEIGHTS = [1.0, 1.0, 0.7]  # ML, DL, Rule-based voting

# === Diverse options til SaaS, API, CI/CD mv. ===
OPTIONS = {
    "telegram_heartbeat_interval": 60,
    "auto_backup": True,
    "use_wandb": False,
    "max_gpus": 1,
}


# ======================================================================
# FASE 4 — PERSISTENS & FILHYGIENE (centrale konstanter og thresholds)
# ======================================================================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if (val is not None and str(val).strip() != "") else default


# Versioner kan overskrives via env (til rollback/migreringer)
SCHEMA_VERSION_DEFAULT = "1.0.0"
FEATURES_VERSION_DEFAULT = datetime.utcnow().strftime("%Y-%m-%d")  # fx "2025-10-11"

PERSIST = {
    # Filrotation: roter når disse tærskler nås (enten/eller)
    "ROTATE_MAX_ROWS": _env_int("PERSIST_ROTATE_MAX_ROWS", 50_000),  # N bars pr. part
    "ROTATE_MAX_MINUTES": _env_int("PERSIST_ROTATE_MAX_MINUTES", 15),  # M minutter pr. part
    # Kompaktering: begræns antallet af små parts pr. dag
    "COMPACT_MAX_PARTS": _env_int("PERSIST_COMPACT_MAX_PARTS", 12),
    # Retention: hvor længe vi beholder ZIP-arkiver (CI-artifacts) og evt. ældre dagsmapper
    "RETENTION_DAYS": _env_int("PERSIST_RETENTION_DAYS", 30),
    # Skema/feature-versionering i Parquet metadata
    "SCHEMA_VERSION": _env_str("PERSIST_SCHEMA_VERSION", SCHEMA_VERSION_DEFAULT),
    "FEATURES_VERSION": _env_str("PERSIST_FEATURES_VERSION", FEATURES_VERSION_DEFAULT),
    # Standardiseret sti-layout (bruges af utils.artifacts)
    # Eksempel: outputs/live/BTCUSDT/1m/date=2025-10-11/part-0001.parquet
    "LAYOUT": {
        "ROOT": OUTPUTS_PATH,
        "LIVE_ROOT": os.path.join(OUTPUTS_PATH, "live"),
        "DAY_PREFIX": "date=",  # dagspartition
        "PART_PREFIX": "part-",  # filprefix
        "PART_EXT": ".parquet",
    },
}


# === Dokumentation ===
"""
- FEATURES: Brug ALL_FEATURES i ML/DL-pipeline, eller vælg grupper dynamisk.
- COINS, TIMEFRAMES: Hentes fra coins_config.py hvis muligt.
- STOP_LOSS, TAKE_PROFIT: Kun fallback. Kan overskrives pr. strategi/coin.
- ML_PARAMS: Brug direkte i træning og tuning.
- TELEGRAM: Sæt credentials via .env.
- OPTIONS: Til SaaS, CI/CD, monitoring, backup mv.
- PERSIST: Central konfiguration til Fase 4 (rotation/kompakt/metadata/layout/retention).
"""


def validate_config() -> bool:
    """Validerer at konfigurationen er konsistent."""
    if not ALL_FEATURES:
        raise ValueError("ALL_FEATURES er tom – mangler featuredefinitioner.")
    if not isinstance(COINS, list) or not COINS:
        raise ValueError("COINS skal være en ikke-tom liste.")
    if not isinstance(TIMEFRAMES, list) or not TIMEFRAMES:
        raise ValueError("TIMEFRAMES skal være en ikke-tom liste.")
    if STOP_LOSS <= 0 or TAKE_PROFIT <= 0:
        raise ValueError("STOP_LOSS og TAKE_PROFIT skal være > 0.")
    _validate_persist()
    return True


def _validate_persist() -> None:
    """Validerer PERSIST-opsætningen (Fase 4)."""
    if PERSIST["ROTATE_MAX_ROWS"] <= 0 and PERSIST["ROTATE_MAX_MINUTES"] <= 0:
        raise ValueError("PERSIST: Mindst én af ROTATE_MAX_ROWS/ROTATE_MAX_MINUTES skal være > 0.")
    if PERSIST["COMPACT_MAX_PARTS"] <= 0:
        raise ValueError("PERSIST: COMPACT_MAX_PARTS skal være > 0.")
    if PERSIST["RETENTION_DAYS"] < 0:
        raise ValueError("PERSIST: RETENTION_DAYS kan ikke være negativ.")
    if not PERSIST["SCHEMA_VERSION"]:
        raise ValueError("PERSIST: SCHEMA_VERSION må ikke være tom.")
    if not PERSIST["FEATURES_VERSION"]:
        raise ValueError("PERSIST: FEATURES_VERSION må ikke være tom.")
    layout = PERSIST.get("LAYOUT", {})
    for key in ("ROOT", "LIVE_ROOT", "DAY_PREFIX", "PART_PREFIX", "PART_EXT"):
        if key not in layout or str(layout[key]).strip() == "":
            raise ValueError(f"PERSIST: LAYOUT.{key} mangler eller er tom.")


# === Eksportér alle variable ===
__all__ = [
    "FEATURES",
    "ALL_FEATURES",
    "COIN_CONFIGS",
    "COINS",
    "TIMEFRAMES",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "RISK_LEVELS",
    "REGIME_THRESHOLDS",
    "DATA_PATH",
    "MODEL_PATH",
    "LOG_PATH",
    "OUTPUTS_PATH",
    "TELEGRAM",
    "ML_PARAMS",
    "ENSEMBLE_WEIGHTS",
    "OPTIONS",
    "PERSIST",
    "validate_config",
]

if __name__ == "__main__":  # lokal sanity
    try:
        if validate_config():
            print("✅ Konfigurationen er gyldig.")
        print(f"Features: {ALL_FEATURES}")
        print(f"Coins: {COINS} | Timeframes: {TIMEFRAMES}")
        print(f"Data path: {DATA_PATH} | Outputs path: {OUTPUTS_PATH}")
        print("Persist:", {k: v for k, v in PERSIST.items() if k != "LAYOUT"})
        print("Layout:", PERSIST["LAYOUT"])
    except Exception as e:
        print(f"❌ Konfigurationsfejl: {e}")
