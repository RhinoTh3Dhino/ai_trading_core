# Observability – AI Trading Core

Denne mappe giver en reproducerbar observability-stack for live_connector:

* Prometheus (metrics, recording rules, alerts)
* Alertmanager (Telegram m.fl.)
* Grafana (auto-provisioneret dashboard: Live Feed – Ops)
* Runbook (PowerShell) til lokal start/stop og promtool tests
* CI (GitHub Actions) med promtool check og unit-tests af alerts

Ikke finansiel rådgivning. Kun til udvikling og driftsovervaagning.

---

## Mappestruktur

```
ops/
  compose/                      docker-compose stack (profiler: default, alerting, ui)
  prometheus/
    prometheus.yml
    recording_rules.yml
    alerts.yml
    tests/alerts_test.yml       promtool unit-tests af alerts
  grafana/
    provisioning/dashboards/dashboards.yml
    dashboards/livefeed.json    "Live Feed – Ops"
  alertmanager/
    render.sh
    alertmanager.runtime.yml
    secrets/telegram_*          (ikke i repo)
runbooks/
  observability.ps1             lokal start/stop/validering
.github/workflows/
  observability.yml             CI pipeline
```

Persistens:

* Prometheus TSDB: named volume `prom_data`
* Grafana state: named volume `grafana_data`

---

## Hurtig start (Windows / PowerShell)

Fra repo-roden:

```powershell
.\ops\runbooks\observability.ps1 -Ui -Alerting -Open
```

Bekraeft:

* App: [http://localhost:8000/healthz](http://localhost:8000/healthz)  ->  {"status":"ok"}
* Prometheus: [http://localhost:9090](http://localhost:9090)  (Status -> Targets = up)
* Grafana: [http://localhost:3000](http://localhost:3000)  (dashboard "Live Feed – Ops")
* Alertmanager: [http://localhost:9093](http://localhost:9093)  (ingen alerts i normal drift)

Stop alt og ryd op:

```powershell
.\ops\runbooks\observability.ps1 -Down
```

macOS/Linux (manuel):

```bash
docker compose -f ops/compose/docker-compose.yml up -d --wait live_connector prometheus
docker compose -f ops/compose/docker-compose.yml --profile alerting up -d am_init alertmanager
docker compose -f ops/compose/docker-compose.yml --profile ui up -d grafana
```

---

## Promtool validering (lokalt)

```powershell
# Check config + rules
docker run --rm `
  -v "$PWD/ops/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" `
  -v "$PWD/ops/prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro" `
  -v "$PWD/ops/prometheus/recording_rules.yml:/etc/prometheus/recording_rules.yml:ro" `
  --entrypoint=promtool prom/prometheus:v3.6.0 `
  check config /etc/prometheus/prometheus.yml

# Enhedstests af alerts
docker run --rm -v "$PWD/ops/prometheus:/etc/prom" `
  --entrypoint=promtool prom/prometheus:v3.6.0 test rules /etc/prom/tests/alerts_test.yml
```

Hvorfor tests:

1. fanger regressions i udtryk, 2) sikrer thresholds og for-vinduer, 3) deterministisk i CI.

---

## Dashboard: Live Feed – Ops (uddrag)

Paneler:

* Transport p99 (ms) pr. venue (histogram)
* Bar close lag (ms) max pr. venue/symbol (5m)
* Bars/sek (global)
* Reconnects (seneste 5m) pr. venue
* Feature latency p95/p99 (ms)
* Queue depth (max/5m)

Variables: `venue`, `symbol`.

---

## Alarmer (uddrag)

* NoBarsIn5m  (ingen bars i 5m pr. venue)
* HighBarCloseLag  (5m maks > 3000 ms pr. venue/symbol)
* TransportLatencyP99High  (p99 > 500 ms pr. venue)
* FeatureLatencyP99High  (p99 > 50 ms pr. feature)
* ReconnectSpike  (increase(feed_reconnects_total[5m]) > 3)
* QueueDepthHigh  (max_over_time(feed_queue_depth[5m]) > 1000)
* Global-varianter for transport og bar-lag

Se fuld liste i `ops/prometheus/alerts.yml`.

---

## Simulation af alarmer (live)

Bem: Live simulation kraever ventetid pga. ratevinduer og for-tider. Promtool tests er hurtigere.

1. NoBarsIn5m

```powershell
docker compose -f ops/compose/docker-compose.yml stop live_connector
# vent 7-8 min (5m rate + 2m for)
```

2. ReconnectSpike

```powershell
for /l %i in (1,1,5) do docker compose -f ops/compose/docker-compose.yml restart live_connector & timeout /t 10 >nul
```

3. QueueDepthHigh
   Skab belastning der giver queue_depth > 1000.
   Til hurtig test kan tærsklen midlertidigt saenkes i alerts.yml (fx > 1) og reloades:

```bash
curl -X POST http://localhost:9090/-/reload
```

Rollback efter test.

4. Transport/Feature p99
   Koer workload der producerer histogram-events. Alternativt saenk tærskler midlertidigt (500 -> 5 ms / 50 -> 5 ms) og reload.

---

## CI pipeline (GitHub Actions)

`.github/workflows/observability.yml`:

* unit-smoke: starter app, checker /metrics, promtool check rules, promtool test rules.
* compose-integration: docker compose, venter paa health, validerer targets via /api/v1/targets, uploader targets.json artifact.

Benefit: tidlig feedback, afviser ændringer der bryder alerts, verificerer scraping i container-miljoe.

---

## Secrets (Alertmanager/Telegram)

Placering (ikke i Git):

```
ops/alertmanager/secrets/telegram_bot_token
ops/alertmanager/secrets/telegram_chat_id
```

Init-job `am_init` bygger `/amcfg/alertmanager.yml` fra template og secrets.

---

## Troubleshooting

Grafana viser tomme paneler:

* Vent et par minutter (scrape-interval + recording-interval).
* Saet time range til Last 6 hours og Refresh 10s.
* Tjek serier i Prometheus (fx feed_bars_total).

Prometheus targets = unknown:

* Normal i foerste scrapes; bliver up efter 10-20s.
* Se logs:

```powershell
docker compose -f ops/compose/docker-compose.yml logs --no-color --tail=200 live_connector
```

Alerts fyrer ikke:

* Husk for-tiden. Brug Explain i Prometheus UI.
* Koer promtool tests (se ovenfor).

Reload regler efter edit:

```bash
curl -X POST http://localhost:9090/-/reload
```

---

## DoD checkliste

* [x] Recording rules loader (promtool OK)
* [x] Minimum 2 alarmer; promtool unit tests PASS
* [x] Dashboard provisioneres og viser baseline
* [x] Persistens for Prometheus og Grafana (named volumes)
* [x] CI groen inkl. promtool test rules
* [x] Runbook starter/stopper stack og printer target-status

---

## Vedligehold

* Pin Prometheus v3.6.0 i compose og CI for determinisme.
* Pin Grafana-tag ved behov.
* Opdater `ops/prometheus/tests/alerts_test.yml` hvis annotation-tekster eller thresholds aendres.
