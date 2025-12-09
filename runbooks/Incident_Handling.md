
---

## `runbooks/Incident_Handling.md`

```markdown
# Incident Handling – AI_TRADING_CORE

## 1. Definitioner

- **Incident (kritisk)**
  Hændelse der kan give tab af kapital, forkert trading eller stoppe kritiske systemer.
- **Major incident**
  Stort tab (fx > 2 %) og/eller længere downtime (> 30 min i tradingtid).

---

## 2. Kategorier

- A – Feed (EPIC A)
- B – Trading/bots
- C – Infrastruktur
- D – Konfiguration
- E – Eksterne systemer (venues, netværk, mm.)

---

## 3. Livscyklus

1. Detektion
2. Stabilisering
3. Analyse & midlertidig fix
4. Post-mortem & læring

---

## 4. Standard procedure ved kritisk incident

### 4.1 Stop trading (fail-safe)

Bot-services ligger i anden compose-fil – brug der den relevante `docker compose`-kommando til at stoppe bot (paper/live).
Princip: stop nye ordrer så hurtigt som muligt.

### 4.2 Stop feed ved datakorruption (hvis relevant)

Kør fra projektroden (tilpas sti efter behov):

```powershell
cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops/compose/docker-compose.yml stop live_connector


Hvis metrics/observability også skal stoppes:

docker compose -f ops/compose/docker-compose.yml stop live_connector prometheus

4.3 Informér

Kort besked i intern kanal:

"Incident på AI_TRADING_CORE – bot/feed stoppet midlertidigt, årsag: <kort beskrivelse>."

4.4 Indsaml evidens

Eksempel:

docker logs live_connector --tail 500 > logs/incident_<dato>_live_connector.log
docker logs prometheus --tail 200 > logs/incident_<dato>_prometheus.log


Gem Grafana-screenshots (feed-latency, DQ, PnL, DD, TE).

4.5 Klassificér hændelse (A–E)

A: feed-issue

B: trading-issue

C: infra-issue

D: config-issue

E: venue/ekstern.

4.6 Midlertidig løsning

Eksempler:

disable problematisk venue / strategi

reducér load/sizing

rulle config tilbage

genstarte services.

4.7 Kontrolleret genstart (feed-siden)

Fra projektroden:

cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops/compose/docker-compose.yml up -d live_connector prometheus
docker compose -f ops/compose/docker-compose.yml --profile ui up -d grafana


Verificér feed-metrics og DQ i Grafana, før bots re-aktiveres.

Så er log-eksport altid robust, også på nye maskiner/ren clone.

if (-not (Test-Path .\logs)) {
    New-Item -ItemType Directory -Path .\logs -Force | Out-Null
}
docker logs live_connector --tail 200 > .\logs\incident_test_live_connector.log
