# config/coins_config.py
"""
Pr-coin & pr-timeframe settings til professionel multi-coin support.
Her styres SL/TP, strategi, aktivering, limits og hvad du vil pr. coin.
"""

COIN_CONFIGS = {
    "BTCUSDT": {
        "enabled": True,
        "sl": 0.02,
        "tp": 0.04,
        "strategy": "ensemble",
        "min_balance": 100,
        "max_position": 0.1,
        "timeframes": ["1h", "4h"],
    },
    "ETHUSDT": {
        "enabled": True,
        "sl": 0.025,
        "tp": 0.05,
        "strategy": "ml",
        "min_balance": 50,
        "max_position": 0.2,
        "timeframes": ["1h", "4h"],
    },
    "DOGEUSDT": {
        "enabled": True,
        "sl": 0.03,
        "tp": 0.06,
        "strategy": "ml",
        "min_balance": 10,
        "max_position": 1.0,
        "timeframes": ["1h"],
    },
    # Tilføj flere coins her ...
}

# Automatisk opsætning af aktiverede coins og timeframes (kun de "enabled")
ENABLED_COINS = [k for k, v in COIN_CONFIGS.items() if v.get("enabled", True)]
ENABLED_TIMEFRAMES = sorted(
    set(
        tf
        for v in COIN_CONFIGS.values()
        if v.get("enabled", True)
        for tf in v.get("timeframes", [])
    )
)

# Eksempel på adgang:
# fra config.coins_config import COIN_CONFIGS
# btc_sl = COIN_CONFIGS["BTCUSDT"]["sl"]
