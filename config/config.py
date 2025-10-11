# config/config.py
# --- Central konfiguration for AI trading bot (modulær & SaaS-ready) ---

import os

# === Feature-grupper ===
FEATURES = {
    "trend": ["ema_9", "ema_21", "ema_50", "ema_200", "macd"],
    "momentum": ["rsi_14", "rsi_28"],  # evt. "stochastic_14"
    "volatility": ["atr_14", "bb_upper", "bb_lower"],
    "volume": ["vwap"],  # evt. "obv"
    "regime": ["adx_14", "zscore_20", "regime"],
    "extra": [],  # tom liste giver ekstra test-coverage
}

# === Automatisk ALL_FEATURES (unik liste, ingen dublikater/kommenterede) ===
ALL_FEATURES = []
for group in FEATURES.values():
    ALL_FEATURES.extend([f for f in group if f and not f.strip().startswith("#")])
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))  # Unique, original rækkefølge

# === Coin/strategi-opsætning fra coins_config.py (fallback hvis ikke fundet) ===
try:
    from config.coins_config import COIN_CONFIGS, ENABLED_COINS, ENABLED_TIMEFRAMES
except ImportError:
    COIN_CONFIGS = {}
    ENABLED_COINS = ["BTCUSDT"]
    ENABLED_TIMEFRAMES = ["1h"]

COINS = ENABLED_COINS
TIMEFRAMES = ENABLED_TIMEFRAMES

# === (Fallback: Default global SL/TP hvis ikke defineret pr. coin) ===
STOP_LOSS = 0.02  # 2%
TAKE_PROFIT = 0.04  # 4%
RISK_LEVELS = [0.01, 0.02, 0.03]

# === Regime-detektion (fx bull/bear thresholds) ===
REGIME_THRESHOLDS = {
    "bull": {"adx_min": 20, "macd_slope": 0.0},
    "bear": {"adx_min": 20, "macd_slope": 0.0},
}

# === Path-opsætning (kan overskrives via env eller CLI) ===
DATA_PATH = os.getenv("DATA_PATH", "data/")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
LOG_PATH = os.getenv("LOG_PATH", "logs/")

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

# === Dokumentation ===
"""
- FEATURES: Brug ALL_FEATURES i ML/DL-pipeline, eller vælg grupper dynamisk.
- COINS, TIMEFRAMES: Hentes fra coins_config.py hvis muligt.
- STOP_LOSS, TAKE_PROFIT: Kun fallback. Kan overskrives pr. strategi/coin.
- ML_PARAMS: Brug direkte i træning og tuning.
- TELEGRAM: Sæt credentials via .env.
- OPTIONS: Til SaaS, CI/CD, monitoring, backup mv.
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
    return True


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
    "TELEGRAM",
    "ML_PARAMS",
    "ENSEMBLE_WEIGHTS",
    "OPTIONS",
    "validate_config",
]

if __name__ == "__main__":
    # CLI-test for hurtig sanity check
    try:
        if validate_config():
            print("✅ Konfigurationen er gyldig.")
        print(f"Features: {ALL_FEATURES}")
        print(f"Coins: {COINS} | Timeframes: {TIMEFRAMES}")
        print(f"Data path: {DATA_PATH}")
    except Exception as e:
        print(f"❌ Konfigurationsfejl: {e}")
