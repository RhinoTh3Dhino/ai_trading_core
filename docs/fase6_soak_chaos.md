# Fase 6 — Soak/Chaos & rapport

Denne run-book beskriver, hvordan Fase 6 køres, overvåges og bruges som release-gate.

## 1. Start observability- og chaos-stack

```bash
# Fra repo-roden
docker compose -f ops/compose/docker-compose.yml up -d live_connector prometheus alertmanager grafana
docker compose -f ops/compose/docker-compose.yml -f ops/compose/docker-compose.chaos.addon.yml --profile chaos up -d chaos_runner
