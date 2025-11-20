# ops/chaos/README.md

# Fase 6 — Soak/Chaos & rapport

## Formål
Valider robusthed under lange runs (soak) og kontrollerede fejl (chaos). Mål SLO’er: MTTR, droputs, reconnect-rate, p50/p95/p99 for transport- og bar-lag.

## Krav/forudsætninger
- Miljø: **STAGE** (ikke PROD) med Docker Compose.
- Prometheus/Grafana/Alertmanager kører.
- `live_connector`-service findes (navn må matche i compose).
- `chaos_runner`-container må køre **privileged** og dele netns med `live_connector`.

## Hurtig kørsel
```bash
# 1) Start Prometheus/Grafana hvis ikke kørende
docker compose -f ops/compose/docker-compose.yml up -d prometheus grafana

# 2) Start chaos-runner (addon compose anbefales, se .chaos.addon.yml)
docker compose -f ops/compose/docker-compose.yml -f ops/compose/docker-compose.chaos.addon.yml --profile chaos up -d chaos_runner

# 3) Start soak (sætter markører og mapper outputs)
bash ops/chaos/scripts/soak_start.sh

# 4) Kør scenarier (eksempler)
bash ops/chaos/scripts/chaos_lib.sh netdrop 20
bash ops/chaos/scripts/chaos_lib.sh throttle 200kbit 100ms 20ms 300
bash ops/chaos/scripts/chaos_lib.sh cpu 60

# 5) Saml data og generér præ-rapport
bash ops/chaos/scripts/soak_collect.sh --final

# 6) Lav PDF/MD-rapport
python ops/chaos/scripts/F6_generate_report.py \
  --metrics-dir outputs/reports/F6_$(date +%Y%m%d_%H%M)/metrics \
  --out-dir     outputs/reports/F6_$(date +%Y%m%d_%H%M)
