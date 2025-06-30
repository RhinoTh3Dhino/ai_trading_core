# config.py
# --- Central konfiguration for din AI trading bot ---

FEATURES = {
    "trend": [
        "ema_9",      # Til EMA-cross/backtest og regime detection!
        "ema_21",
        "ema_50",     # Bruges til Bollinger Bands
        "ema_200",
        "macd"
    ],
    "momentum": [
        "rsi_14",
        "rsi_28",
        # "stochastic_14",   # Fjern evt. fra tests indtil implementeret!
    ],
    "volatility": [
        "atr_14",
        "bb_upper",
        "bb_lower"
    ],
    "volume": [
        # "obv",        # (KAN KOMMENTERES UD indtil implementeret – ellers fejler tests!)
        "vwap"
    ],
    "regime": [
        "adx_14",
        "zscore_20",
        "regime"   # (label, sættes automatisk af pipelinen)
    ],
    "extra": [
        # "supertrend_10_3", # Fjern eller kommentér ud indtil implementeret!
        # "cci_20"
    ]
}

COINS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
TIMEFRAMES = ["1h", "4h"]

STOP_LOSS = 0.02      # 2%
TAKE_PROFIT = 0.04    # 4%
RISK_LEVELS = [0.01, 0.02, 0.03]  # Til position sizing

REGIME_THRESHOLDS = {
    "bull": {"adx_min": 20, "macd_slope": 0.0},
    "bear": {"adx_min": 20, "macd_slope": 0.0}
}

DATA_PATH = "data/"
MODEL_PATH = "models/"
LOG_PATH = "logs/"
TELEGRAM = {
    "token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID"
}

# (Tilføj flere SaaS-options, API-keys, ML-parametre osv. efter behov)
