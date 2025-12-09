# V1 Scope – Lyra-TradeOps (Beta)

## Formål

Definere et klart, realistisk scope for Lyra V1 (Beta), så udvikling, homelab og drift arbejder efter samme ramme.

## Overordnet V1-mål

- Én primær venue: **Binance (spot)**.
- Én gennemtestet **flagship-strategi**.
- Fuld kæde: **data → backtest → paper → UI → homelab-drift**.
- Systemet kører stabilt som **paper trading** 24/7 i homelab.

## In scope for V1 (Beta)

- **Data & feed**
  - Binance REST/WS connector (spot, testnet/prod).
  - Historik til TimescaleDB.
  - Simpelt feature-lag til flagship-strategi.

- **Backtest & paritet**
  - Backtest-pipeline for flagship-strategi.
  - Fill-engine v2 (partials, latency, simpel impact).
  - Replay-dag parity (TE-måling backtest vs paper).

- **Paper trading & risk (EPIC C – V1 light)**
  - Execution-adapter til Binance testnet.
  - Pre-trade risk-engine (max position, max exposure, dags-DD-limit).
  - Daglig TE-rapport mellem backtest og paper.

- **UI v1**
  - Dashboard med PnL, positions, status.
  - Start/stop af bot (inkl. SAFE_STOP).
  - Simpel trade-log.

- **CI/CD & observability**
  - Lint + pytest + basic backtest/replay-job i CI.
  - Prometheus + Grafana + (evt.) Alertmanager i homelab.
  - Basis-runbooks til start/stop, backup/restore.

- **Homelab (Proxmox)**
  - Single-node Proxmox.
  - VM `lyra-core` med hele stacken (DB + bot + monitoring).
  - Regelmæssig backup og dokumenteret restore.

## Out of scope (efter V1)

Disse elementer parkeres bevidst til **efter** V1 Beta:

- Multi-venue feed orchestrator (hot-standby, auto-failover).
- Options/Black-Scholes engine og options-GUI.
- ML drift-detektion og auto-retrain.
- Kelly/adaptiv position sizing.
- Avanceret Telegram-kommando-suite.
- Proxmox-cluster med flere noder og HA.

## V1 KPI’er (Beta-niveau)

Foreløbige Beta-KPI’er (kan justeres senere):

- **Backtest/OOS:** Sharpe og DD på niveau med strategi-krav (defineres i strategi-dokument).
- **Paritet:** Median TE (backtest vs paper) ≤ ca. 10 % for flagship på Binance.
- **Paper-drift:** ≥ 99 % uptime i paper trading i homelab (trading hours).
- **Stabilitet:** Ingen ukendte, gentagne crashes uden dokumenteret root cause.
- **Homelab:** Backup/restore testet mindst én gang og dokumenteret.
