6. Periodisk restore-test

Mindst 1 gang/mdr:

Spind testmiljø op.

Restore DB + MLflow.

Verificér at systemerne kan køre på restored data.

Dokumentér testen.


---

## `runbooks/Paper_And_Live_Bots.md`

*(ingen ændring nødvendig ift. `\`, men vi undgår Linux-line-continuation og holder kommandoer enkle – stadig generiske for trading-compose)*

```markdown
# Paper- og Live-bots – Drift & Kontrol

## 1. Formål

Beskrive drift af:

- paper-bots (STAGE)
- live-bots (PROD)

og relationen til LiveFeed.

---

## 2. Afhængighed til LiveFeed

Bots må ikke køre på ustabilt feed.
Hvis LiveFeed har kritiske problemer → stop bots og stabilisér feed først.

---

## 3. Paper-bot – start/stop (eksempel)

Antag en separat compose-fil for trading, fx `ops\compose\trading\docker-compose.yml`.

Start paper-bot:

```powershell
cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops\compose\trading\docker-compose.yml --profile paper up -d bot timescaledb mlflow gui


Stop paper-bot:

docker compose -f ops\compose\trading\docker-compose.yml --profile paper stop bot


Tilpas filnavn/profiler når din trading-compose er på plads.

4. Live-bot – start/stop (eksempel)

Forudsætninger: OOS- og paper-KPI’er opfyldt, LiveFeed stabil.

Start live-bot:

cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops\compose\trading\docker-compose.yml --profile live up -d bot gui


Stop live-bot:

docker compose -f ops\compose\trading\docker-compose.yml --profile live stop bot

5. RefStrat_01 (EMA/ATR BTCUSDT 1h)

Overvåg tæt:

Sharpe (rolling)

Max DD

Trades pr. periode

TE vs. backtest/paper.

6. Escalation-regler

Risk-limits brudt → stop live-bot, følg Incident_Handling.md.

Feed degraderet → stop live-bot/paper-bot, fix feed (LiveFeed_Runbook.md).

TE/paritet kraftigt skæv → pause deploys, genanalysér fill-engine/strategi/feed.

7. Logging og dokumentation

Opdater strategi-dokumentation ved større ændringer.

Registrér større sizing/strategi-shifts i CHANGELOG eller separat drift-log.


---

Hvis du vil, kan vi tage næste skridt og lave en konkret `ops\compose\trading\docker-compose.yml` til
