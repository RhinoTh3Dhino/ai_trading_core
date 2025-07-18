# config/monitoring_config.py
"""
Konfiguration af overvågning, alarmer og monitoring-regler for AI trading bot.
Ændr her – og alt i botten følger med!
"""

# === Overordnede aktiveringer ===
ENABLE_MONITORING = True      # Slå alt monitoring til/fra globalt

# === Thresholds/grænser for alarmer (kan redigeres) ===
ALARM_THRESHOLDS = {
    "winrate": 25,       # Minimum win-rate i % (fx 25 = 25%) før alarm
    "drawdown": -20,     # Maks drawdown i % (fx -20 betyder alarm ved -20%)
    "profit": -10,       # Maks negativ profit i % (fx -10 = -10%)
}

# === Individuel alarm-aktivering ===
ALERT_ON_WINRATE = True       # Telegram-alarm hvis win-rate < threshold
ALERT_ON_DRAWNDOWN = True     # Telegram-alarm hvis drawdown < threshold
ALERT_ON_PROFIT = True        # Telegram-alarm hvis profit < threshold

# === Ekstra alarmer og betingelser ===
ALERT_ON_CONSEC_LOSSES = False     # Alarm ved X tab i træk
CONSEC_LOSSES_THRESHOLD = 8        # Fx alarm ved >8 tab i træk

# === Telegram-format og præcision ===
TELEGRAM_METRIC_PRECISION = 2      # Antal decimaler i Telegram-metrics

# === Notifikation og status-frekvens ===
NOTIFY_FREQUENCY_MINUTES = 60      # Send status/max én gang pr. time

# === Multi-symbol/multi-timeframe settings ===
ENABLED_SYMBOLS = None             # Fx ["BTCUSDT", "ETHUSDT"] eller None for alle
ENABLED_TIMEFRAMES = None          # Fx ["1h", "4h"] eller None for alle

# === Monitoring paths og default-parametre for live-simulering ===
LIVE_SIM_FEATURES_PATH = "outputs/feature_data/live_features.csv"
LIVE_SIM_INITIAL_BALANCE = 1000
LIVE_SIM_NROWS = 300
LIVE_SIM_CHAT_ID = None
MODEL_TYPE = "ML"
LIVE_SIM_SYMBOL = "btcusdt"
LIVE_SIM_TIMEFRAME = "1h"
LIVE_SIM_FEATURES_DIR = "outputs/feature_data"

# === Version, navngivning og udvidelser ===
MONITORING_VERSION = "v1.0.1"

# === Fremtidig: webhook, e-mail, sms, alert-matrix etc. ===
# ALERT_WEBHOOK_URL = None
# ENABLE_EMAIL_ALERTS = False

# === Whitelist/blacklist for coins/timeframes (ekstra granularitet) ===
# BLOCKED_SYMBOLS = []
# BLOCKED_TIMEFRAMES = []

# === Placeholder for fremtidig config-integration ===

