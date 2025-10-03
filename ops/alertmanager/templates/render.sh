#!/bin/sh
set -euo pipefail

echo "[am_init] start – secrets > env"
# secrets -> env
[ -r /run/secrets/telegram_bot_token ] && export TELEGRAM_BOT_TOKEN="$(cat /run/secrets/telegram_bot_token)"
[ -r /run/secrets/telegram_chat_id ]  && export TELEGRAM_CHAT_ID="$(cat /run/secrets/telegram_chat_id)"

# valider
: "${TELEGRAM_BOT_TOKEN:?missing TELEGRAM_BOT_TOKEN}"
: "${TELEGRAM_CHAT_ID:?missing TELEGRAM_CHAT_ID}"
: "${ALERT_TMPL:?missing ALERT_TMPL}"
: "${ALERT_OUT:?missing ALERT_OUT}"

mkdir -p "$(dirname "$ALERT_OUT")"
envsubst '${TELEGRAM_BOT_TOKEN} ${TELEGRAM_CHAT_ID}' <"$ALERT_TMPL" >"$ALERT_OUT"

echo "[am_init] wrote $ALERT_OUT"
sed -n '1,40p' "$ALERT_OUT" || true
ls -l "$(dirname "$ALERT_OUT")" || true
