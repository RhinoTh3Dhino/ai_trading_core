# Roadmap – Lyra V1 (Beta) Sprints

## Sprint 0 – Plan & oprydning

**Mål:** Fastlægge V1-scope og få `ai_bot_dev` i ren, grøn tilstand.

- Opret V1_SCOPE.md.
- Opret ROADMAP_V1_SPRINTS.md.
- Sikre tests/lint/CI er grøn på `ai_bot_dev`.
- Opret GitHub Project og EPIC-issues.

## Sprint 1 – EPIC C V1 kerne (Paper + Risk + TE)

**Mål:** Minimal robust paper-kæde for flagship-strategi på Binance testnet.

- Execution-adapter til Binance testnet (market/limit).
- Pre-trade risk-engine (max pos, max exposure, dags-DD).
- TE-script: backtest vs paper for valgte dage.

## Sprint 2 – Homelab-drift & observability

**Mål:** Få paper-kæden til at køre stabilt i homelab.

- Deploy dev-stack på Proxmox VM `lyra-core`.
- TimescaleDB + bot + Prometheus + Grafana.
- 24–72 timers sammenhængende paper-run i homelab.

## Sprint 3 – EPIC B paritet & backtest-kvalitet

**Mål:** Fornuftig paritet mellem backtest og paper.

- Fill-engine v2 færdig og testet.
- Replay-day parity-job med TE-rapport.
- Første seriøse OOS-rapport for flagship-strategi.

## Sprint 4 – EPIC E UI v1 (kontrolpanel & status)

**Mål:** Gøre systemet brugbart uden terminal.

- API-lag (status, PnL, trades, start/stop).
- UI v1 med PnL, positions, status, kontrol.
- UI-deploy i homelab.

## Sprint 5 – Beta-hardening & bruger-klar pakke

**Mål:** Gøre systemet stabilt nok til at kalde det “Beta”.

- Runbooks for homelab og drift.
- 30 dages stabil paper-drift (samlet).
- Beta-statusrapport + kendte begrænsninger.
