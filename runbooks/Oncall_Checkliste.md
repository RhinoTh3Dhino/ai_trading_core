7. Relation til andre runbooks

Incidents: Incident_Handling.md

Deploy/rollback: Deploy_Rollback.md

Backup/restore: Backup_And_Restore.md


---

## `runbooks/Oncall_Checkliste.md`

```markdown
# On-call checkliste (AI_TRADING_CORE)

## 1. Før du går on-call

Sikre:

- Adgang til server/homelab (SSH/VPN).
- Adgang til Grafana, Prometheus, Alertmanager, MLflow.
- Adgang til GitHub-repo (`AI_TRADING_CORE`).
- Du kan:
  - køre `docker ps`, `docker logs`, `docker compose ...`
  - læse/rette `config/` og `runbooks/`.

---

## 2. Start af vagt (daglig standardrutine)

### 2.1 Systemstatus

1. Åbn PowerShell og gå til projektroden:

   ```powershell
   cd C:\Users\reno_\Desktop\ai_trading_core


Tjek containere:

docker ps


live_connector bør være Up (eller bevidst stoppet).

prometheus og grafana bør være Up, når observability er aktiv.

Alerting og debug efter behov.

Tjek Grafana “System Health”-dashboard:

CPU/RAM/disk OK.

Ingen nye voldsomme error-spikes.

2.2 LiveFeed (EPIC A)

I Grafana (Feed-dashboard):

feed_transport_latency_ms p99 < 500 ms.

feed_bar_close_lag_ms stabil.

feed_bars_total stiger (rate > 0 for aktive symbols).

feed_reconnects_total ikke eksploderer.

DQ-paneler (gaps_total, nan_records_total) tæt på 0.

2.3 Trading-bots (paper/live)

I Trading-dashboard:

Equity-kurve uden uventede spikes.

Dagens PnL og max DD inden for risk-limits.

Antal trades pr. periode realistisk.

2.4 ML-drift (EPIC G)

Drift-rapport (PSI) for nøglefeatures:

PSI < 0.2 som udgangspunkt.

Evt. auto-retrain events gennemgås for sanity.

2.5 Alertmanager

Gennemgå åbne alerts.

Ingen uadresserede kritiske alerts.

3. Under vagt

Ansvar:

Reagere på alarmer.

Sikre:

feed stabilt

bots indenfor risk-limits

incidents stabiliseres og dokumenteres.

Du skal:

bringe system i sikker tilstand

oprette issues/post-mortems

eskalere ved behov.

4. Slut på vagt (overlevering)

Kort status til næste on-call:

incidents (ID, tid, kort beskrivelse)

feed-status

bot-status (PnL, DD, TE, risk-limits).

Opdater runbooks hvis nye mønstre/løsninger.

Sikre kritiske issues er oprettet i backlog.

5. Kommando-cheatsheet (LiveFeed/Observability)

Kør fra projektroden, fx:

cd C:\Users\reno_\Desktop\ai_trading_core


Start LiveFeed + Prometheus:

docker compose -f ops/compose/docker-compose.yml up -d live_connector prometheus


Start fuld observability (LiveFeed + Prometheus + Grafana):

docker compose -f ops/compose/docker-compose.yml --profile ui up -d grafana


Start alerting-stack:

docker compose -f ops/compose/docker-compose.yml --profile alerting up am_init
docker compose -f ops/compose/docker-compose.yml --profile alerting up -d alertmanager


Stop LiveFeed:

docker compose -f ops/compose/docker-compose.yml stop live_connector
