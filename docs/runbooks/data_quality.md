# Runbook — Data Quality

## Alarmer
- **DQViolationsBurst**: Åbn Prometheus → “dq_violations_total”. Se labels {contract, rule}.
- **DataFreshnessStale**: Bekræft ingestion-job, netværk, credentials, diskplads.
- **GlobalNullRateHigh**: Kør lokalt: `python tools/check_data_quality.py --dataset <sti> --contract ohlcv_1h --print-report`.

## Lokal reproduktion
1) Hent seneste fil til `outputs/...`
2) Kør CLI (se ovenfor) og læs JSON issues.
3) Fix upstream (schema, typer, nulls) og reprocessér.

## Ejerskab
- Primary: Live Connector / Data pipeline
- Secondary: ML/Backtest (afhænger af dataset)



# Runbook — Data Quality

## Symptomer
- Alert **DataFreshnessStale** (kritisk): `dq_freshness_minutes{dataset="..."}` > threshold i > for-vindue.
- Alert **DQViolationsBurst** (warning): `increase(dq_violations_total[5m]) > 0`.

## Quick checks
```bash
# Targets up?
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[].health'

# Freshness / Violations
curl -s "http://localhost:9090/api/v1/query?query=dq_freshness_minutes"
curl -s "http://localhost:9090/api/v1/query?query=sum by (contract,rule) (increase(dq_violations_total[5m]))"



## v0.5.0 — Fase 5: Datakvalitet & alerts
### Added
- FastAPI prod-endpoints: `/dq/freshness`, `/dq/violation` (auth via `X-Dq-Secret`)
- Data ingestion: `freshness_reporter.py` hook efter write/rotation
- Data quality: `checks.py` (global_null_rate → `dq_violations_total`)
- Prometheus: `rules/data_quality.yml` + CI promtool checks
- Grafana: `Data Quality & Live Latency` dashboard (provisioning)
- Alertmanager→Telegram: templated runtime config + `am_init` renderer
- Runbook: `docs/runbooks/data_quality.md`

### Changed
- `docker-compose`: faste volumen-navne, healthchecks, profiler
- `observability.yml` workflow: metrics smoke, rules lint, compose-integration

### Security
- `/dq/*` kræver `DQ_SHARED_SECRET` i header `X-Dq-Secret`.
