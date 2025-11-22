5. Post-mortem skabelon

reports/incidents/INC_<YYYYMMDD>_<kort_navn>.md:

Titel

Dato/tid

Kategori (A–E)

Detektion

Impact

Timeline

Root cause

Midlertidig fix

Permanent fix

Prevention

Læring / action items


---

## `runbooks/Deploy_Rollback.md`

```markdown
# Deploy & Rollback – AI_TRADING_CORE

## 1. Formål

Standardiseret deploy- og rollback-flow for:

- LiveFeed-stack (EPIC A) via `ops/compose/docker-compose.yml`
- Trading-core stack (bots, DB, MLflow, GUI) via anden compose-fil (ikke beskrevet her i detaljer).

---

## 2. LiveFeed – Deploy

### 2.1 Forudsætninger

- Kode for `live_connector` opdateret og committed.
- CI-tests grønne.
- `LIVE_ENV_FILE` sat korrekt (lokalt: `../../config/env/live.env`).

### 2.2 Byg og deploy LiveFeed lokalt

Fra projektroden:

```powershell
cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops/compose/docker-compose.yml build live_connector
docker compose -f ops/compose/docker-compose.yml up -d live_connector prometheus
docker compose -f ops/compose/docker-compose.yml --profile ui up -d grafana

2.3 Alerting-stack
cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops/compose/docker-compose.yml --profile alerting up am_init
docker compose -f ops/compose/docker-compose.yml --profile alerting up -d alertmanager

3. LiveFeed – Rollback

Da live_connector bruger build + image: live-connector:local, sker rollback via Git:

Rul kode tilbage:

cd C:\Users\reno_\Desktop\ai_trading_core
git checkout <stable-commit>


Rebuild + start:

docker compose -f ops/compose/docker-compose.yml build live_connector
docker compose -f ops/compose/docker-compose.yml up -d live_connector prometheus
docker compose -f ops/compose/docker-compose.yml --profile ui up -d grafana


Verificér i Grafana at feed ser normal ud.
