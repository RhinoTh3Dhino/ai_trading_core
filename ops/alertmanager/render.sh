#!/bin/sh
# Robust Alertmanager-config renderer (Alpine/BusyBox-kompatibel)

# Fail fast (POSIX): -e (exit on error), -u (unset vars error)
# pipefail er ikke POSIX i ash; prøv, men ignorér fejl hvis ikke understøttet
set -eu
# shellcheck disable=SC3040
set -o pipefail 2>/dev/null || true

die() { echo "[am_init] ERROR: $*" >&2; exit 2; }

echo "[am_init] start – læser secrets og renderer template"

# 0) Værktøj: envsubst (fra gettext). Vi er i Alpine, så apk er tilgængelig.
if ! command -v envsubst >/dev/null 2>&1; then
  echo "[am_init] installing gettext (envsubst)…"
  apk add --no-cache gettext >/dev/null 2>&1 || die "kunne ikke installere gettext"
fi

# 1) Kræv stier/vars (POSIX-parameter expansion)
: "${ALERT_TMPL:?ALERT_TMPL ikke sat (sti til template)}"
: "${ALERT_OUT:?ALERT_OUT ikke sat (sti til genereret config)}"

# 2) Læs secrets fra Docker secrets (rå)
[ -r /run/secrets/telegram_bot_token ] || die "secret telegram_bot_token mangler"
[ -r /run/secrets/telegram_chat_id ]   || die "secret telegram_chat_id mangler"

# 3) Trim evt. CR (\r) fra Windows-encoding
# BusyBox 'tr' er tilgængelig
TELEGRAM_BOT_TOKEN="$(tr -d '\r' </run/secrets/telegram_bot_token)"
TELEGRAM_CHAT_ID="$(tr -d '\r' </run/secrets/telegram_chat_id)"

# 4) Let sanity: ikke-tomme værdier
[ -n "$TELEGRAM_BOT_TOKEN" ] || die "TELEGRAM_BOT_TOKEN er tom"
[ -n "$TELEGRAM_CHAT_ID" ]   || die "TELEGRAM_CHAT_ID er tom"

# 5) Eksportér kun de variabler vi vil substituere
export TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID

# 6) Render template → output (brug envsubst med $VAR-filtrering, ikke ${VAR})
outdir="$(dirname -- "$ALERT_OUT")"
mkdir -p "$outdir"
# Bemærk: envsubst filtrerer kun de nævnte variabler
envsubst '$TELEGRAM_BOT_TOKEN $TELEGRAM_CHAT_ID' <"$ALERT_TMPL" >"$ALERT_OUT".tmp

# 7) Sanity: tjek for uerstattede placeholders af typen ${...}
# Brug ERE ( -E ) og + for "en eller flere"
if grep -Eq '\$\{[A-Za-z0-9_]+\}' "$ALERT_OUT".tmp; then
  echo "[am_init] WARNING: fundet uerstattede placeholders i output:"
  grep -En '\$\{[A-Za-z0-9_]+\}' "$ALERT_OUT".tmp || true
fi

# 8) Sanity: fil må ikke være tom
if [ ! -s "$ALERT_OUT".tmp ]; then
  rm -f "$ALERT_OUT".tmp
  die "genereret fil er tom – check template/vars"
fi

# 9) Atomisk move ind på plads
mv -f "$ALERT_OUT".tmp "$ALERT_OUT"

# 10) Debug-udskrift (maskér længder, ikke indhold)
# BusyBox wc -c giver med mellemrum – cut for at få kun tallet
token_len="$(printf %s "$TELEGRAM_BOT_TOKEN" | wc -c | awk '{print $1}')"
chat_len="$(printf %s "$TELEGRAM_CHAT_ID" | wc -c | awk '{print $1}')"
echo "[am_init] token_len=${token_len} chat_len=${chat_len}"
echo "[am_init] skrev $ALERT_OUT (første 40 linjer):"
sed -n '1,40p' "$ALERT_OUT" || true

echo "[am_init] ls -l /amcfg:"
ls -l /amcfg || true

echo "[am_init] done ✅"
