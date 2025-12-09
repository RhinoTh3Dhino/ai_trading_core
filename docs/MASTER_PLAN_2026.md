# Lyra-TradeOps – Masterplan 2025–2026

## 1. Vision & målbillede

**Vision:**
Lyra-TradeOps er en modulær, professionel AI trading-platform med fokus på:

- Robust, realistisk backtest
- Stabil live/paper execution
- Gennemsigtig risiko og drift
- Klar vej til SaaS og 10 betalende power users ved udgangen af 2026

**Målbillede 2026:**

- 1–2 gennemtestede strategier i produktion
- Min. 10 betalende power users på en stabil V1
- Homelab/Proxmox fungerer som primær produktions- og testplatform
- Klar arkitektur til LLM/ML-udvidelser (4090-node, drift-detektion, options)

---

## 2. Overordnet roadmap

### 2.1 Faser

1. **Fase A – V1 Beta Core (NU → ~3–6 mdr.)**
   - Én exchange (Binance)
   - Én flagship-strategi
   - Fuldt flow: data → backtest → paper → UI → homelab-drift

2. **Fase B – V1.0 Pro (~6–12 mdr.)**
   - Hardening, multi-miljø (dev/test/pro)
   - Bedre observability, backup/restore, onboarding
   - 3–5 betalende brugere

3. **Fase C – Udvidelse & LLM (2026)**
   - Drift-detektion, adaptiv sizing (Kelly)
   - LLM-node til feature-engineering/analyse
   - Skalering til 10+ power users

---

## 3. V1-scope

### 3.1 In scope (V1)

- **Én venue:** Binance (spot, testnet + senere prod)
- **Én flagship-strategi:** klart dokumenteret, backtestet og paper-kørt
- **Data & feed:**
  - Binance REST/WS connector
  - Historik til TimescaleDB
  - Simpelt feature-lag til flagship-strategi
- **Backtest & paritet:**
  - Fill-engine v2 (partials, latency, basic impact)
  - Replay-dag paritetstest (TE ≤ 10 %)
- **Paper trading:**
  - Execution-adapter til Binance testnet
  - Pre-trade risk-engine (pos-caps, DD-limit, circuit-breaker)
- **UI v1:**
  - PnL, positions, trades, feed-status
  - Start/stop af bot
- **CI/CD & observability:**
  - Lint, pytest, basic replay-job
  - Prometheus + Grafana + Alertmanager
  - Telegram-notifikationer
- **Homelab:**
  - Proxmox-node
  - Lyra-core VM med Docker/Compose
  - Backup & simpel driftsrunbook

### 3.2 Out of scope (efter V1)

- Multi-venue feed orchestrator (hot-standby, auto-switch)
- Options/Black-Scholes engine og GUI
- ML drift-detektion og auto-retrain
- Kelly-baseret sizing
- Avanceret Telegram-suite
- Større Proxmox-cluster (2–3 noder)

---

## 4. Sprints til V1

### Sprint 0 – Alignment & oprydning

**Mål:** Fælles ramme, rent repo, V1 defineret.

**Tjekliste:**

- [ ] Opret `docs/V1_SCOPE.md` (indhold svarende til afsnit 3)
- [ ] Opret `docs/ROADMAP_V1_SPRINTS.md` med Sprint 0–7
- [ ] Ryd op i `ai_bot_dev` (tests, lint, CI grøn)
- [ ] Opret GitHub Project: “Lyra V1 & Beta 2026”
- [ ] Opret epic-issues (A, B, C, E, F, HOMELAB, PRODUCT)
- [ ] Aftal simple KPI’er for V1 (Sharpe, TE, uptime, FEED-uptime)

---

### Sprint 1 – Flagship-strategi & backtest

**Mål:** Flagship-strategi + robust backtest-pipeline.

**Tjekliste:**

- [ ] Dokumentér flagship-strategi:
  - [ ] `docs/strategy/FLAGSHIP_BINANCE.md`
  - [ ] Parametre, tidsramme, univers
- [ ] Implementér backtest-pipeline:
  - [ ] `bot/backtest/strategy_flagship_binance.py`
  - [ ] `scripts/backtest_flagship_binance.py`
- [ ] Implementér fill-engine v2:
  - [ ] `bot/backtest/fill_engine.py`
  - [ ] `tests/backtest/test_fill_engine_flagship.py`
- [ ] Generér første OOS-rapport (lokalt)

**Exit-kriterier:**

- [ ] Reproducerbar backtest for flagship
- [ ] Fill-engine dækket af tests
- [ ] Enkel OOS-rapport (CSV/HTML)

---

### Sprint 2 – Homelab & miljøer

**Mål:** Lyra stack kører i homelab i DEV-miljø.

**Tjekliste:**

- [ ] Opret VM `lyra-core` i Proxmox (Ubuntu/Debian)
- [ ] Installer Docker, Compose, git, Python
- [ ] Clone `ai_trading_core` → `~/projects/ai_trading_core_dev`
- [ ] Checkout `ai_bot_dev` og kør `pytest`
- [ ] Konfigurer `.env.dev` og secrets (lokalt)
- [ ] Start dev-stack i homelab:
  - [ ] TimescaleDB
  - [ ] Lyra-bot (dev)
  - [ ] Prometheus + Grafana (basic)
- [ ] Verificér adgang:
  - [ ] Grafana UI
  - [ ] Simple metrics for bot

**Exit-kriterier:**

- [ ] Boten kan køre i homelab (demo/soak)
- [ ] Basis-metrics tilgængelige i Grafana

---

### Sprint 3 – UI v1 (status & kontrol)

**Mål:** Minimal UI til monitoring og kontrol.

**Tjekliste:**

- [ ] Definér UI-krav i `docs/UI_V1_REQUIREMENTS.md`
- [ ] Implementér API-endpoints:
  - [ ] PnL, positions, åbne ordrer
  - [ ] Status (feed ok, risk ok)
  - [ ] Start/stop
- [ ] Implementér UI (fx Streamlit/React):
  - [ ] Forside med PnL & eksponering
  - [ ] Signal/trade-feed
  - [ ] Start/stop-knap
- [ ] Deploy UI i homelab (bag reverse proxy eller direkte port)

**Exit-kriterier:**

- [ ] UI kan bruges til at starte/stoppe bot
- [ ] Bruger ser PnL, exposures og status live

---

### Sprint 4 – Onboarding & API-nøgler

**Mål:** Forberede onboarding af første eksterne test-brugere.

**Tjekliste:**

- [ ] Definér API-key håndtering:
  - [ ] Hvor gemmes nøgler (fil, DB, vault)
  - [ ] Hvordan mappe dem til brugere
- [ ] Opret `docs/ONBOARDING_CHECKLIST.md`:
  - [ ] Hvilke oplysninger kræves fra bruger
  - [ ] Steps til opsætning
- [ ] Implementér simpel bruger-konfiguration (YAML/DB)
- [ ] Test onboarding-flow med dummy-bruger

**Exit-kriterier:**

- [ ] Én fiktiv bruger kan sættes op fra nul til kørende bot efter checklisten

---

### Sprint 5 – Risk & SAFE_STOP

**Mål:** Risiko styring og sikker stop.

**Tjekliste:**

- [ ] Implementér pre-trade risk-engine:
  - [ ] Max position per symbol
  - [ ] Max samlet eksponering
  - [ ] Dags-Drawdown limit
- [ ] Implementér SAFE_STOP:
  - [ ] Knappen i UI
  - [ ] Telegram-kommando (`/safe_stop`)
- [ ] Logik for auto-stop ved:
  - [ ] DD > grænse
  - [ ] For mange ordrefejl
- [ ] Tests for risk-engine og SAFE_STOP

**Exit-kriterier:**

- [ ] Risk-levels håndhæves automatisk
- [ ] Bot stopper sikkert på kommando + ved brud på limits

---

### Sprint 6 – Beta-program & basic billing

**Mål:** Klar til at håndtere de første betalende brugere.

**Tjekliste:**

- [ ] Definér beta-program:
  - [ ] Målgruppe
  - [ ] Kontrakt/T&C udkast
  - [ ] Support-kanal (fx privat Telegram-gruppe)
- [ ] Simpelt billing-setup (kan være manuelt til at starte med):
  - [ ] Prissætning
  - [ ] Faktura-/betaling-proces
- [ ] Opret `docs/BETA_PROGRAM.md`

**Exit-kriterier:**

- [ ] Minimum-setup til at tage imod 1–3 betalende power users

---

### Sprint 7 – Hardening & V1-beta launch

**Mål:** Færdiggøre V1 Beta og køre 24/7 paper-run i homelab.

**Tjekliste:**

- [ ] Soak-test (min. 30 dages paper-run i homelab)
- [ ] Review af logs og TE-metrics
- [ ] Opdater runbooks:
  - [ ] Start/stop, deployment
  - [ ] Fejlfinding (feed, DB, API)
- [ ] Endelig V1-beta checklist:
  - [ ] Data/Feed
  - [ ] Backtest/Paritet
  - [ ] Paper/Execution
  - [ ] UI
  - [ ] Risk
  - [ ] Homelab/Drift
- [ ] Vælg 3–5 kandidater til beta-program og klargør onboarding

**Exit-kriterier:**

- [ ] 24/7 paper trading uden kritiske fejl
- [ ] Klar til at sætte de første eksterne brugere på

---

## 5. Homelab-plan (Proxmox)

### Fase 1 – Single-node (NU)

- [ ] Proxmox host opsat og patchet
- [ ] VM `lyra-core` (DB + bot + monitoring)
- [ ] Proxmox-backup til ekstern disk/NAS
- [ ] UPS opsat og testet
- [ ] Dokumentation: `docs/HOMELAB_SETUP.md`

### Fase 2 – Split af workloads (V1+)

- [ ] Opret VM `lyra-db` (TimescaleDB + evt. Prometheus)
- [ ] Opret VM `lyra-core` (bots, API, UI, MLflow)
- [ ] Opret VM `lyra-ops` (Grafana, Alertmanager, n8n, logging)
- [ ] Flyt services til de nye VM’er
- [ ] Opdater monitoring & backup-plan

### Fase 3 – Cluster (efter 10+ brugere)

- [ ] Vurder behov for 2–3 Proxmox-noder
- [ ] Design cluster-arkitektur (HA, shared storage)
- [ ] Implementér og dokumentér

---

## 6. LLM- og 4090-node

### Mål

- Adskille trading-core fra tunge LLM/ML workloads
- Bruger 4090-node til research, feature-engineering og analyse

### Tjekliste

- [ ] Definér LLM-anvendelser (feature-engineering, sentiment, regime)
- [ ] Beslut cloud vs lokal 4090-node (tid, pris, latency)
- [ ] Hvis lokal:
  - [ ] Installer Linux + Docker på 4090-node
  - [ ] Opret intern API-service (HTTP) til LLM-kald
  - [ ] Dokumentér integration med Lyra (`ml/llm_client.py`)
- [ ] Sæt simple tests og benchmarks op

---

## 7. Brug af AI-agenter

### Mål

- Øge output (kode, tests, docs) via systematisk brug af LLM’er

### Tjekliste

- [ ] Design “Code-review agent”-prompt (Claude/GPT)
- [ ] Design “Test-generator agent”-prompt
- [ ] Design “Runbook/doc-agent”-prompt
- [ ] Indfør rutine:
  - [ ] Alle kritiske PR’s gennemgår code-review agent
  - [ ] Nye moduler får test-forslag fra test-agent
  - [ ] Nye services får runbook via doc-agent

---

## 8. KPI’er og Go/No-Go

### Kerne-KPI’er (V1)

- [ ] Backtest Sharpe (OOS) ≥ defineret minimum
- [ ] Replay TE median ≤ 10 %
- [ ] Paper uptime ≥ 99 % (trading hours)
- [ ] Kritiske fejl < defineret threshold per måned
- [ ] Homelab-backups verificeret min. månedsvis

---

## 9. Risici & mitigering (oversigt)

- **Over-kompleksitet før V1**
  - Mitigering: hård V1-scope, ingen nye epics før V1-sprints er færdige
- **Homelab-nedetid**
  - Mitigering: UPS, backups, runbooks
- **Single-dev flaskehals**
  - Mitigering: AI-agenter til kode, tests og docs; fokus på vigtigste tasks
- **Regulatoriske og exchange-ændringer**
  - Mitigering: modulær connector-arkitektur, daglig/ugentlig review af API-ændringer

---

_End of MASTER_PLAN_2026_
