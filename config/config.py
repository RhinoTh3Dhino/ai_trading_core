# config/config.py
# --- Central konfiguration for din AI trading bot (modulær og SaaS-ready) ---

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
        # "obv",
        "vwap"
    ],
    "regime": [
        "adx_14", "zscore_20", "regime"
    ],
    "extra": [
        # "supertrend_10_3",
        # "cci_20",
    ]
}

# === Automatisk ALL_FEATURES (ingen dublikater) ===
ALL_FEATURES = []
for group in FEATURES.values():
    ALL_FEATURES.extend([f for f in group if not f.startswith("#") and f.strip() != ""])
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))  # Unik liste

# === Trading-opsætning ===
COINS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
TIMEFRAMES = ["1h", "4h"]

STOP_LOSS = 0.02      # 2%
TAKE_PROFIT = 0.04    # 4%
RISK_LEVELS = [0.01, 0.02, 0.03]  # Til position sizing

# === Regime-detektion (f.eks. bull/bear) thresholds ===
REGIME_THRESHOLDS = {
    "bull": {"adx_min": 20, "macd_slope": 0.0},
    "bear": {"adx_min": 20, "macd_slope": 0.0}
}

# === Path-opsætning (kan overskrives i env eller CLI) ===
DATA_PATH = os.getenv("DATA_PATH", "data/")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
LOG_PATH = os.getenv("LOG_PATH", "logs/")

# === Telegram/Notification (brug .env til prod, kun dummy default her) ===
TELEGRAM = {
    "token": os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
}

# === ML/DL modelparametre (kan udbygges) ===
ML_PARAMS = {
    "batch_size": 64,
    "epochs": 30,
    "seq_length": 48,
    "learning_rate": 0.001,
    # Tilføj flere parametre her!
}

# === Ensemble/strategi-weights ===
ENSEMBLE_WEIGHTS = [1.0, 1.0, 0.7]  # ML, DL, Rule

# === Diverse options til SaaS, API, CI/CD mv. ===
OPTIONS = {
    "telegram_heartbeat_interval": 60,     # minutter
    "auto_backup": True,
    "use_wandb": False,
    "max_gpus": 1,
    # ...
}

# === Dokumentation ===
"""
- FEATURES: Brug ALL_FEATURES i ML/DL-pipeline, eller vælg grupper dynamisk.
- COINS, TIMEFRAMES: Automatisk support for multi-coin/-timeframe backtest og trading.
- STOP_LOSS, TAKE_PROFIT: Standard risk management.
- ML_PARAMS: Kan bruges direkte i model-træning og hyperparam tuning.
- TELEGRAM: Skal sættes via .env for sikkerhed i produktion!
- OPTIONS: Til SaaS, CI/CD, cloud, monitoring, backup mv.
"""

# === Nem adgang fra andre moduler ===
__all__ = [
    "FEATURES", "ALL_FEATURES", "COINS", "TIMEFRAMES", "STOP_LOSS", "TAKE_PROFIT",
    "RISK_LEVELS", "REGIME_THRESHOLDS", "DATA_PATH", "MODEL_PATH", "LOG_PATH",
    "TELEGRAM", "ML_PARAMS", "ENSEMBLE_WEIGHTS", "OPTIONS"
]
