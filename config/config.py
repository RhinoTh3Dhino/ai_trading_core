# config.py
# --- Central konfiguration for din AI trading bot ---

# Valgte tekniske indikatorer/features (let udvidet så ALLE tests/CI passer!)
FEATURES = {
    "trend": [
        "ema_9",      # <- Tilføjet så alle tests går igennem!
        "ema_21",
        "ema_50",     # <- Typisk brugt til Bollinger Bands
        "ema_200",
        "macd"
    ],
    "momentum": [
        "rsi_14",
        "rsi_28",
        "stochastic_14"   # Du kan evt. implementere denne senere
    ],
    "volatility": [
        "atr_14",
        "bb_upper",
        "bb_lower"
    ],
    "volume": [
        "obv",        # Hvis ikke implementeret endnu, kan du midlertidigt fjerne fra test
        "vwap"
    ],
    "regime": [
        "adx_14",
        "zscore_20",
        "regime"      # Bruges ofte til label/markering
    ],
    "extra": [
        "supertrend_10_3", # Kan implementeres senere eller fjernes fra test
        "cci_20"
    ]
}

# Coins og timeframes (let at udvide til multi-asset/multi-timeframe)
COINS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
TIMEFRAMES = ["1h", "4h"]

# Parametre for risk management og strategi
STOP_LOSS = 0.02      # 2%
TAKE_PROFIT = 0.04    # 4%
RISK_LEVELS = [0.01, 0.02, 0.03]  # fx til dynamic position sizing

# Regime/feature parametre (kan tilføjes efter behov)
REGIME_THRESHOLDS = {
    "bull": {"adx_min": 20, "macd_slope": 0.0},
    "bear": {"adx_min": 20, "macd_slope": 0.0}
}

# Pipeline-options
DATA_PATH = "data/"
MODEL_PATH = "models/"
LOG_PATH = "logs/"
TELEGRAM = {
    "token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID"
}

# Evt. flere SaaS-options, API-keys, ML-parametre osv.
