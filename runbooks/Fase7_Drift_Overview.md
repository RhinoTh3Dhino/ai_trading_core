# Fase 7 – Drift & Runbooks (AI_TRADING_CORE)

## Formål

Denne runbook beskriver, hvordan `AI_TRADING_CORE` og tilhørende services drives i daglig drift:

- LiveFeed (EPIC A – live datafeed & streaming-features)
- Trading-bots (paper/live)
- Underliggende platform (TimescaleDB, MLflow, Prometheus, Grafana)
- CI/CD, backup, incidents og on-call

Målet er, at en teknisk person kan:

- starte/stoppe systemerne kontrolleret
- overvåge health via dashboards og metrics
- reagere korrekt på alarmer
- udføre simple deploy/rollback og restore.

---

## 1. Relation til Beta-sprintplan (v3)

Driftlagets runbooks understøtter især:

- **EPIC A – Live datafeed & streaming-features**
  Fase 0–7 (latency, features, multi-venue, persistens, DQ, soak/chaos, drift & runbooks).
- **EPIC B – Backtest realisme & paritet**
- **EPIC C – Paper trading robusthed & execution**
- **EPIC F – CI/CD, observability, accounting & docs**
- **EPIC G/H/T – ML drift-detection, adaptive sizing, avanceret Telegram**

Runbooks = operationalisering af disse epics – ikke design-dokumenter.

---

## 2. Miljøer og hovedkomponenter

### 2.1 Miljøer

- **DEV (lokalt / homelab)**
  Ad hoc-tests, udvikling. Kørsel via `docker compose` og lokal Python.
- **STAGE / PAPER**
  Kontinuerlig test med paper trading.
- **PROD / LIVE**
  Rigtige penge, lav eksponering, stramme risk-limits og alarmer.

### 2.2 Kerne-services

- `live_connector` – LiveFeed (EPIC A).
- `prometheus` – metrics.
- `grafana` – dashboards.
- `alertmanager` + `am_init` – alarmer.
- (I anden compose-fil): `bot`, `timescaledb`, `mlflow`, `gui` osv.

---

## 3. Driftprincipper

1. **Feed før bot**
   Bots må ikke køre på ustabilt/korrupt feed.
2. **Reference-strategi (RefStrat_01)**
   EMA/ATR BTCUSDT 1h bruges som reference ift. backtest/paritet.
3. **Observability før skalering**
   Ingen produktion uden `/metrics`, dashboards og alerts.
4. **No silent failure**
   Kritiske fejl → logs + dashboards + Alertmanager.
5. **Fail-safe ved tvivl**
   Ukendt tilstand → stop nye ordrer (og evt. bot-container).

---

## 4. Daglig drift (high-level)

### 4.1 Daglig on-call rutine

Se detaljer i `Oncall_Checkliste.md`. Overordnet:

1. Systemstatus:
   - `docker ps` → alle kritiske services `Up`.
2. LiveFeed:
   - latency, `bars_total`, reconnects, DQ-metrics ok.
3. Trading-bots:
   - equity, DD, PnL og TE indenfor limits.
4. ML-drift:
   - drift-rapport (PSI) ok; auto-retrain under kontrol.
5. Alerts:
   - ingen uadresserede kritiske alerts.

---

## 5. Ugentlige/månedlige rutiner

- Review OOS/TE-rapport og incidents.
- Review drift/PSI-rapporter.
- Test backup-restore jævnligt (mindst 1 gang/mdr i testmiljø).

---

## 6. Relaterede runbooks

- `LiveFeed_Runbook.md`
- `Oncall_Checkliste.md`
- `Incident_Handling.md`
- `Deploy_Rollback.md`
- `Backup_And_Restore.md`
- `Paper_And_Live_Bots.md`

Disse udgør tilsammen Fase 7 – Drift & Runbooks for AI_TRADING_CORE.
