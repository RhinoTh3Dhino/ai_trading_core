runbooks/LiveFeed_Runbook.md
# LiveFeed Runbook (EPIC A – Live datafeed & streaming-features)

## 1. Formål

Drift af **LiveFeed-servicen** fra EPIC A:

- multi-venue WebSocket/REST feed
- streaming-features (EMA/RSI/VWAP/ATR m.m.)
- persistens (Parquet/DB, når koblet på)
- observability (Prometheus/Grafana)
- datakvalitet og alerts

Mål: opfylde **EPIC A Fase 0–7**, især Fase 7: “let at køre, let at fejlfinde”.

---

## 2. Service-overblik

Defineret i `ops/compose/docker-compose.yml`:

- `live_connector` – hovedfeed (`bot.live_connector.cli`).
- `prometheus` – scraper `live_connector:8000/metrics`.
- `grafana` – UI (profil: `ui`).
- `am_init` + `alertmanager` – alerting (profil: `alerting`).
- `sample_emitter`, `promtool_check` – debug (profil: `debug`).

---

## 3. CLI & ENV (overordnet)

`live_connector` starter med:

```yaml
command:
  - python
  - -m
  - bot.live_connector.cli
  - --quiet
env_file:
  - ${LIVE_ENV_FILE:-../../config/env/live.env}
environment:
  PYTHONUNBUFFERED: "1"
  LOG_LEVEL: "INFO"


Typiske env-variabler i config/env/live.env (eksempler, afhængigt af implementation):

VENUES_ENABLED=binance,bybit,okx,kraken

SYMBOLS=BTCUSDT,ETHUSDT

INTERVALS=1m,1h

QUIET=true (eller via --quiet)

STATUS_MIN_SECS=60

PARTITIONING_ENABLED=true

Hold runbook og live.env i sync.

4. Start/stop via Docker Compose (PowerShell-eksempler)

Kør alle kommandoer fra projektroden, fx:

cd C:\Users\reno_\Desktop\ai_trading_core

4.1 Start kun LiveFeed + Prometheus
docker compose -f ops/compose/docker-compose.yml up -d live_connector prometheus

4.2 Start fuld observability-stack (LiveFeed + Prometheus + Grafana)
docker compose -f ops/compose/docker-compose.yml --profile ui up -d grafana


Denne ene kommando starter automatisk:

live_connector

prometheus

grafana

pga. depends_on.

4.3 Start alerting-stack (Alertmanager)
docker compose -f ops/compose/docker-compose.yml --profile alerting up am_init
docker compose -f ops/compose/docker-compose.yml --profile alerting up -d alertmanager


Evt. se init-log:

docker compose -f ops/compose/docker-compose.yml --profile alerting logs -n 60 am_init

4.4 Start debug-profiler (valgfrit)
docker compose -f ops/compose/docker-compose.yml --profile debug up -d sample_emitter promtool_check

4.5 Stop LiveFeed og observability

Stop kun LiveFeed:

docker compose -f ops/compose/docker-compose.yml stop live_connector


Stop LiveFeed + Prometheus:

docker compose -f ops/compose/docker-compose.yml stop live_connector prometheus


Stop Grafana (UI):

docker compose -f ops/compose/docker-compose.yml --profile ui stop grafana


Stop Alertmanager:

docker compose -f ops/compose/docker-compose.yml --profile alerting stop alertmanager


Stop debug-services:

docker compose -f ops/compose/docker-compose.yml --profile debug stop sample_emitter promtool_check

5. Metrics, dashboards og alarmer (kort)

Nøgle-metrics (via /metrics på live_connector):

feed_transport_latency_ms{venue,symbol}

feed_bar_close_lag_ms{venue,symbol}

feed_bars_total{venue,symbol}

feed_reconnects_total{venue}

feed_gaps_total{venue,symbol}

feed_nan_records_total{venue,symbol}

feature_compute_ms{feature,symbol}

Dashboards i Grafana:

Feed Overview (latency, bars/sec, reconnects).

Data Quality (gaps, NaN, cross-venue divergence).

Alerts i Alertmanager:

stalled feed

høj latency

stigende DQ-fejl.

6. Common errors & quick actions

Eksempler:

Stalled feed

Symptom: feed_bars_total flader ud; alert “feed stalled”.

Handling:

docker logs live_connector --tail 200

evt. docker compose -f ops/compose/docker-compose.yml stop live_connector

ret ENV/bug

docker compose -f ops/compose/docker-compose.yml up -d live_connector.

Høj latency

Symptom: p99 feed_transport_latency_ms > 500 ms.

Handling:

netværkstjek

reducer load (symbols/venues)

overvåg i Grafana.

Mange NaN/gaps

Handling:

undersøg venue vs. parser

evt. disable venue midlertidigt

fix parser og redeploy.
