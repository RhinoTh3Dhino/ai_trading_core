````markdown
# Runbook — Fase 5: Datakvalitet & alerts

> **Formål:** Sikre at live-data er friske, konsistente og alarmeres korrekt. Denne runbook dækker signaler, triage, afhjælpning, verifikation og ansvar.

---

## 1) Omfang & definitioner

**Datasæt (eksempel):** `ohlcv_1h`
**Kerne-metrikker:**
- `dq_freshness_minutes{dataset=...}` — minutter siden seneste vellykkede opdatering (Gauge)
- `dq_violations_total{contract=...,rule=...}` — antal registrerede DQ-brud (Counter)
- Live-feed støtte: `feed_transport_latency_ms_*`, `feed_bar_close_lag_ms`, `feed_bars_total`, `feed_reconnects_total`

**SLO’er (anbefaling):**
- Freshness (ohlcv_1h): **≤ 15 min** i **99%** af tiden pr. dag
- DQ-violations: **0** burst-violations i normaldrift (tolerer enkelte i retraining-vinduer)

---

## 2) Signaler & alarmer

**Aktive alerter (Prometheus rules):**
- **DataFreshnessStale (critical)**
  `dq_freshness_minutes{dataset="ohlcv_1h"} > 15` for 5m
  _Indikation:_ Ingestion job hænger/fejler eller upstream er nede

- **DQViolationsBurst (warning)**
  `increase(dq_violations_total[5m]) > 0` for 2m
  _Indikation:_ Kontraktovertrædelser (skema, bounds, NaN-rate, spikes m.v.)

**Dashboards (Grafana):**
“**Data Quality & Live Latency**” viser:
- Freshness-gauge pr. dataset
- Violations pr. kontrakt/rule
- Transport-latency & bar-lag (early symptom for feed/clock drift)
- Reconnect-rate (net/venue issues)

---

## 3) Hurtig triage (decision tree)

1) **Alert type?**
   - _Freshness_ → Gå til §4.A
   - _Violations_ → Gå til §4.B

2) **Bekræft målretning**
   Åbn Prometheus **instant queries**:
```bash
# Overblik
curl -s "http://localhost:9090/api/v1/query?query=dq_freshness_minutes"
curl -s "http://localhost:9090/api/v1/query?query=sum by (contract,rule) (increase(dq_violations_total[15m]))"
````

3. **Korriger & verificér**
   Efter fix → kør verifikation i §5.

---

## 4) Standard playbooks

### A) **DataFreshnessStale (critical)**

**Symptom:** `dq_freshness_minutes{dataset="ohlcv_1h"}` overstiger threshold og stabiliserer sig ikke.

**Tjek (i rækkefølge):**

1. **Service health**

   * `GET http://localhost:8000/healthz` → OK?
   * `GET http://localhost:8000/ready` → `{"ready": true}`? (ellers kig på `lag_ms`)
2. **Prometheus target**

   * `GET http://localhost:9090/api/v1/targets` → `live_connector` skal være **up**
3. **App-log & compose**

   * `docker compose -f ops/compose/docker-compose.yml logs --tail=200 live_connector`
4. **Ingestion job/hook**

   * Bekræft at pipeline skriver filer **og** at post-write hook kalder `/dq/freshness`:

     * Eksempel kald:
       `POST /dq/freshness?dataset=ohlcv_1h&minutes=<0..15>` med header `X-Dq-Secret`

**Typiske årsager & afhjælpning:**

* **Ingestion fejlede**: rett fejl (cred/net/kvoter), re-run job, *post* freshness.
* **Hook mangler**: tilføj/ret hook i `utils/dq_wiring.py` eller ingest-scriptet.
* **Clock drift**: synk host-tid (`chrony`/`timesyncd`) → freshness beregnes mod “nu”.

**Manuel normalisering (midlertidigt):**

```bash
curl -s -X POST \
  -H "X-Dq-Secret: $DQ_SHARED_SECRET" \
  "http://localhost:8000/dq/freshness?dataset=ohlcv_1h&minutes=5"
```

---

### B) **DQViolationsBurst (warning)**

**Symptom:** En eller flere regler brydes i løbet af de seneste minutter.

**Identificér præcist brud:**

```bash
# Top violations seneste 15 min
curl -s "http://localhost:9090/api/v1/query?query=topk(10,sum by (contract,rule) (increase(dq_violations_total[15m])))"
```

**Lokal reproduktion (CLI):**

```bash
python tools/check_data_quality.py \
  --dataset outputs/data/btc_1h_latest.parquet \
  --contract ohlcv_1h \
  --print-report --fail-on-issues
```

**Klassiske regler & fixes:**

* `schema_missing` → skemaændring upstream; tilpas ETL og/eller `features_version`
* `nan_rate_excess` → udfyld/forward-fill konservativt, eller kassér part; find root-cause
* `bounds_min_price`/`bounds_min_volume` → filtrér “dust”/negativ volumen; valider parse
* `returns_extreme_spikes` → de-spike strategi (winsorize/clip) med audit-log

**Post til connector (for at trigge alarm bevidst i test):**

```bash
curl -s -X POST \
  -H "X-Dq-Secret: $DQ_SHARED_SECRET" \
  "http://localhost:8000/dq/violation?contract=ohlcv_1h&rule=bounds_min_price&n=3"
```

---

## 5) Verifikation efter fix

1. **Metrics på appen:**

```bash
curl -s http://localhost:8000/metrics | grep -E "dq_freshness_minutes|dq_violations_total" | head
```

2. **Prometheus (instant):**

```bash
# Freshness skal falde til <= threshold
curl -s "http://localhost:9090/api/v1/query?query=dq_freshness_minutes{dataset=\"ohlcv_1h\"}"

# Violations bør ikke stige yderligere
curl -s "http://localhost:9090/api/v1/query?query=sum by (contract,rule) (increase(dq_violations_total[5m]))"
```

3. **Grafana**: Dashboardet “Data Quality & Live Latency” skal vise normaliserede værdier.

---

## 6) CI/Observability pipelines

**Workflows:**

* `.github/workflows/observability.yml`

  * **Unit-smoke:** Starter `uvicorn`, poster freshness, assert’er metrikker
  * **Promtool:** Linter rules og config
  * **Compose-integration:** Kører `live_connector + prometheus`, verificerer targets & queries

**Typiske CI-fejl & løsninger:**

* `/_debug/emit_sample → 403` → workflow skal sætte `ENABLE_DEBUG_ROUTES=1` i **unit-smoke**-jobbet (er gjort i skabelonen).
* Prometheus query 400 → vent til Prometheus er ready og/eller at metrikken eksisterer. I compose-jobbet anvendes **instant query** efter posting.

---

## 7) Ejerskab & kontaktpunkter

* **Primary:** Data/Feed & Ingestion (Live Connector team)
* **Secondary:** ML/Backtest (afhængig af datasæt), Platform/Observability (Prometheus/Alertmanager/Grafana)

---

## 8) Parametre & thresholds (opsummering)

| Parameter                     | Standard              | Placering                                       |
| ----------------------------- | --------------------- | ----------------------------------------------- |
| Freshness-grænse (`ohlcv_1h`) | 15 min                | `rules/data_quality.yml` → `DataFreshnessStale` |
| DQ burst-vindue               | 5 min                 | `rules/data_quality.yml` → `DQViolationsBurst`  |
| Secret header til `/dq/*`     | `X-Dq-Secret`         | `runner.py` + env `DQ_SHARED_SECRET`            |
| Debug routes on/off           | `ENABLE_DEBUG_ROUTES` | `observability.yml` (unit-smoke), compose env   |
| Bootstrap af metrics          | `METRICS_BOOTSTRAP=1` | `metrics.py` (auto ved import)                  |

---

## 9) Reference — API & CLI

**Connector API (prod-endpoints):**

* `POST /dq/freshness?dataset=ohlcv_1h&minutes=NN` (header `X-Dq-Secret`)
* `POST /dq/violation?contract=ohlcv_1h&rule=bounds_min_price&n=1` (header `X-Dq-Secret`)

**Lokal CLI (ad-hoc):**

```bash
python tools/check_data_quality.py \
  --dataset <path.parquet|csv> \
  --contract ohlcv_1h \
  --print-report \
  [--post-endpoint http://localhost:8000 --post-secret $DQ_SHARED_SECRET] \
  [--fail-on-issues]
```

---

## 10) Fejlfinding (udvidet)

**A. “Freshness falder aldrig”**

* Confirm at ingestion faktisk skriver ny part (filstørrelse/mtime)
* Confirm post-hook kører (logs) og returnerer **200 OK**
* Tjek tidsstempel-kilde i ingestion (UTC/epoch ms vs s)
* Check container-ur (`date -u`) og host-NTP

**B. “Violations forsvinder ikke”**

* Counters er monotone; brug rate/increase i dashboards
* Rens/korrektion i dataset fjerner ikke historiske counts, men nyt spring udebliver

**C. “Promtool fejler i CI”**

```bash
docker run --rm \
  -v "$PWD/ops/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  --entrypoint=promtool prom/prometheus:v3.6.0 \
  check config /etc/prometheus/prometheus.yml
```

---

## 11) Artefakter & placeringer

* **Rules:** `ops/prometheus/rules/data_quality.yml`
* **Prom config:** `ops/prometheus/prometheus.yml`, `ops/prometheus/alerts.yml`, `ops/prometheus/recording_rules.yml`
* **Alertmanager:** `ops/alertmanager/` (+ `am_init` renderer)
* **Connector:** `bot/live_connector/runner.py`, `bot/live_connector/metrics.py`
* **CLI:** `tools/check_data_quality.py`
* **Workflows:** `.github/workflows/observability.yml`, `connector-ci.yml`, `ci.yml`
* **Compose:** `ops/compose/docker-compose.yml`

---

## 12) Checkliste ved ændringer (DoD)

* [ ] Nye DQ-regler tilføjet i **Prometheus rules** og promtool-lint’et
* [ ] Dashboard opdateret med relevante paneler/queries
* [ ] CLI understøtter ny regel/kontrakt (hvis relevant)
* [ ] Runbook opdateret (denne fil)
* [ ] Observability-workflow passerer på PR

---

## 13) Bilag — Hurtige PowerShell-eksempler (Windows)

```powershell
# Freshness post
$env:DQ_SHARED_SECRET="change-me-long-random"
iwr -UseBasicParsing -Method POST `
  -Headers @{ "X-Dq-Secret" = $env:DQ_SHARED_SECRET } `
  "http://localhost:8000/dq/freshness?dataset=ohlcv_1h&minutes=5"

# Prometheus query
iwr -UseBasicParsing "http://localhost:9090/api/v1/query?query=dq_freshness_minutes{dataset=`"ohlcv_1h`"}"
```

---

**Status:** *Fase 5* fuldt implementeret: prod-endpoints, metrikker, regler, dashboards, Alertmanager, CI-lint + compose-integration, samt denne runbook. Brug denne som første reference ved alle DQ-alarmer.

```
```
