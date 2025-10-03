#!/bin/sh
set -euo pipefail

# Sikr envsubst findes
apk add --no-cache gettext >/dev/null 2>&1 || true

echo "[am_init] start – læser secrets og renderer template"

# 1) Læs secrets (rå)
[ -r /run/secrets/telegram_bot_token ] || { echo "[am_init] ERROR: secret telegram_bot_token mangler"; exit 2; }
[ -r /run/secrets/telegram_chat_id ]   || { echo "[am_init] ERROR: secret telegram_chat_id mangler"; exit 2; }

# Trim evt. CR (\r) fra Windows-encoding
TELEGRAM_BOT_TOKEN="$(tr -d '\r' </run/secrets/telegram_bot_token)"
TELEGRAM_CHAT_ID="$(tr -d '\r' </run/secrets/telegram_chat_id)"

# 2) Debug-længder
echo "[am_init] token_len=$(printf %s "$TELEGRAM_BOT_TOKEN" | wc -c) chat_len=$(printf %s "$TELEGRAM_CHAT_ID" | wc -c)"

# 3) Gør dem til ENV-variabler (NØGLEN)
export TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID

# 4) Debug at de ER i env
echo "[am_init] env TELEGRAM_*:"
env | grep -E '^TELEGRAM_(BOT_TOKEN|CHAT_ID)=' || true

# 5) Stier fra env
: "${ALERT_TMPL:?ALERT_TMPL ikke sat}"
: "${ALERT_OUT:?ALERT_OUT ikke sat}"

# 6) Render KUN disse placeholders (brug ${VAR}-navne i ENKLE QUOTES)
mkdir -p "$(dirname "$ALERT_OUT")"
envsubst '${TELEGRAM_BOT_TOKEN} ${TELEGRAM_CHAT_ID}' <"$ALERT_TMPL" >"$ALERT_OUT"

echo "[am_init] skrev $ALERT_OUT (første 40 linjer):"
sed -n '1,40p' "$ALERT_OUT" || true

echo "[am_init] ls -l /amcfg:"
ls -l /amcfg || true
