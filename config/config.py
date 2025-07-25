# config/config.py
# --- Central konfiguration for AI trading bot (modulær & SaaS-ready) ---

import os

# === Feature-grupper ===
FEATURES = {
    "trend": [
        "ema_9", "ema_21", "ema_50", "ema_200", "macd"
    ],
    "momentum": [
        "rsi_14", "rsi_28",
        # "stochastic_14",
    ],
    "volatility": [
        "atr_14", "bb_upper", "bb_lower"
    ],
    "volume": [
        "vwap"
        # "obv",
    ],
    "regime": [
        "adx_14", "zscore_20", "regime"
    ],
    "extra": [
        # "supertrend_10_3",
        # "cci_20",
    ]
}

# === Automatisk ALL_FEATURES (unik liste, ingen dublikater/kommenterede) ===
ALL_FEATURES = []
for group in FEATURES.values():
    ALL_FEATURES.extend([f for f in group if not f.strip().startswith("#") and f.strip() != ""])
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
STOP_LOSS = 0.02      # 2%
TAKE_PROFIT = 0.04    # 4%
RISK_LEVELS = [0.01, 0.02, 0.03]  # Til position sizing og gridsearch

# === Regime-detektion (fx bull/bear thresholds) ===
REGIME_THRESHOLDS = {
    "bull": {"adx_min": 20, "macd_slope": 0.0},
    "bear": {"adx_min": 20, "macd_slope": 0.0}
}

# === Path-opsætning (kan overskrives via env eller CLI) ===
DATA_PATH = os.getenv("DATA_PATH", "data/")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
LOG_PATH = os.getenv("LOG_PATH", "logs/")

# === Telegram/Notification (sæt rigtige værdier via .env i produktion) ===
TELEGRAM = {
    "token": os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
}

# === ML/DL modelparametre (kan udbygges løbende) ===
ML_PARAMS = {
    "batch_size": 64,
    "epochs": 30,
    "seq_length": 48,
    "learning_rate": 0.001,
    # Tilføj flere parametre her!
}

# === Ensemble/strategi-weights ===
ENSEMBLE_WEIGHTS = [1.0, 1.0, 0.7]  # ML, DL, Rule-based voting

# === Diverse options til SaaS, API, CI/CD mv. ===
OPTIONS = {
    "telegram_heartbeat_interval": 60,     # minutter
    "auto_backup": True,
    "use_wandb": False,
    "max_gpus": 1,
    # ...
}

# === Dokumentation (kort guide til brug og standarder) ===
"""
- FEATURES: Brug ALL_FEATURES i ML/DL-pipeline, eller vælg grupper dynamisk.
- COINS, TIMEFRAMES: Hentes fra coins_config.py hvis muligt.
- STOP_LOSS, TAKE_PROFIT: Kun fallback. Kan overskrives pr. strategi/coin.
- ML_PARAMS: Brug direkte i træning og tuning.
- TELEGRAM: Sæt credentials via .env.
- OPTIONS: Til SaaS, CI/CD, monitoring, backup mv.
"""

# === Eksportér alle variable til nem import fra andre moduler ===
__all__ = [
    "FEATURES", "ALL_FEATURES", "COIN_CONFIGS", "COINS", "TIMEFRAMES",
    "STOP_LOSS", "TAKE_PROFIT", "RISK_LEVELS", "REGIME_THRESHOLDS",
    "DATA_PATH", "MODEL_PATH", "LOG_PATH", "TELEGRAM", "ML_PARAMS",
    "ENSEMBLE_WEIGHTS", "OPTIONS"
]
