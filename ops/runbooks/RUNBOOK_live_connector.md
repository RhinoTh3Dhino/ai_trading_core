# RUNBOOK – Live Connector & Observability

## 1. Formål
Live Connector streamer live markedsdata, eksponerer `/metrics` til Prometheus og sender kritiske alarmer via Alertmanager → Telegram.

---

## 2. Start/stop

### Start hele observability-stacken (live + alerting + UI)
```powershell
cd C:\Users\reno_\Desktop\ai_trading_core
docker compose -f ops/compose/docker-compose.yml `
  --profile live `
  --profile alerting `
  --profile ui `
  up -d


## Stop

docker compose -f ops/compose/docker-compose.yml down --remove-orphans
