# config.py
# --- Central konfiguration for din AI trading bot ---

# Valgte tekniske indikatorer/features (kan let udvides)
FEATURES = {
    "trend": ["ema_21", "ema_200", "macd"],
    "momentum": ["rsi_14", "stochastic_14"],
    "volatility": ["atr_14", "bb_upper", "bb_lower"],
    "volume": ["obv", "vwap"],
    "regime": ["adx_14", "zscore_20"],
    "extra": ["supertrend_10_3", "cci_20"]
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
