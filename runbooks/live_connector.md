# Runbook – Live Connector & Observability

## 1. Formål

Denne runbook beskriver **drift af live connectoren** og tilhørende observability-stack:

- `live_connector` (FastAPI + metrics)
- `prometheus`
- `alertmanager`
- `grafana`
- (valgfrit) `sample_emitter`, `promtool_check`

Mål: En ny person skal kunne **starte/stoppe, verificere driftstilstand og reagere på alarmer** uden hjælp.

---

## 2. Komponentoversigt

| Service         | Port (host) | Rolle                                      |
|----------------|------------:|--------------------------------------------|
| live_connector | 8000        | Live datafeed, /metrics, debug-routes      |
| prometheus     | 9090        | Scraper metrics, evaluerer rules           |
| alertmanager   | 9093        | Håndterer alarmer, sender Telegram-alerts  |
| grafana        | 3000        | Dashboards og visualisering                |
| am_init        | –           | Init-job, renderer alertmanager.yml        |
| sample_emitter | –           | Dev/debug: genererer test-metrics          |
| promtool_check | –           | Dev/debug: validerer Prometheus rules      |

---

## 3. Konfiguration (ENV & CLI)

### 3.1. ENV-fil

Live-connectoren læser primært konfiguration fra:

- `config/env/live.env`

Nøgler (uddrag):

- `LIVE_VENUES` – aktiverede venues (fx `BINANCE,BYBIT`).
- `LIVE_SYMBOLS` – symbols (fx `BTCUSDT,ETHUSDT`).
- `LIVE_INTERVAL` – bar-interval (fx `1m`).
- `LIVE_QUIET` – `true` / `false`.
- `LIVE_STATUS_MIN_SECS` – min. sekunder mellem statuslinjer.
- `LIVE_OUTPUT_ROOT` – rodmappe for Parquet-output.
- `LIVE_PARTITIONING_ENABLED` – `true` / `false`.

### 3.2. CLI

CLI’en hedder `bot.live_connector.cli` og kaldes enten:

- **Lokalt (venv aktiv):**

  ```bash
  python -m bot.live_connector.cli --quiet
