# Sprint 0–6: Plan + Go/No-Go gates

## Sprint 0 — Governance & gates
Gates:
- Templates + PR-template committed
- CI minimum bar aktiv
- Milestones oprettet

## Sprint 1 — CORE contracts + engine happy path
Gates:
- Engine kan køre end-to-end (paper/replay) uden exceptions
- Contracts serialiseringstest grøn

## Sprint 2 — Data kvalitet + features v1 + baseline strategi
Gates:
- Gap/outlier policy virker + testet
- Feature determinisme (snapshot) grøn
- Baseline strategi producerer signal stabilt

## Sprint 3 — Backtest + risk + logging
Gates:
- No-lookahead backtest test grøn
- Risk kill switch kan stoppe sikkert
- Structured logs med run-id

## Sprint 4 — Execution hardening
Gates:
- Idempotens + ledger + reconciliation
- Testnet smoke test (integration) grøn

## Sprint 5 — Ops baseline
Gates:
- compose baseline “grøn”
- /health + /metrics virker
- Min. 1 alert testet end-to-end

## Sprint 6 — Minimal API + UI
Gates:
- Browser-baseret status + start/stop/SAFE_STOP
- Auth + rate limits + basic permissions
