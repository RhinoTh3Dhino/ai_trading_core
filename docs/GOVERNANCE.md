# Governance

Formål: Skabe en enkel, bindende proces for udvikling og drift mod en stabil Beta.

## 1) Styringsprincipper
- Source of truth: GitHub Issues + Milestones + dokumenter i `docs/`.
- Minimer scope creep: Sprint gates styrer hvad der er “færdigt”.
- Reproducerbarhed først: determinisme, tests og logning går forud for profit-optimering.

## 2) Issue-typer og forventninger
- type:epic: Samler stories for et sprint/outcome.
- type:story: Konkrete leverancer i kode/docs/tests.
- type:bug: Fejl med repro steps + fix + regression test.
- type:chore: Vedligehold/infra/governance.

Alle issues skal have:
- Klar titel (prefix fx `DATA-201`, `BT-401` osv.)
- AC/DoD (eller reference til `docs/Definition_of_Done.md`)
- Milestone (Sprint X)
- Labels: prioritet + område

## 3) Sprint cadence
- Sprint-længde: 1–2 uger (fast beslutning i teamet).
- Hver sprint har:
  - 1 epic (styringscontainer)
  - 3–7 stories (leverancer)
  - 2–5 Go/No-Go gates (se `docs/SPRINTS_0_6.md`)

## 4) Go/No-Go gates
- Gates er bindende og dokumenteres i `docs/SPRINTS_0_6.md`.
- Hvis en gate ikke kan nås:
  - Skal der være en eksplicit beslutning i epics kommentar (hvad udskydes og hvorfor).
  - Scope skal reduceres, ikke “skubbes ind” i sprintet.

## 5) Branch- og merge-flow
Standard flow:
- Arbejdsbranch: `feat/*`, `fix/*`, `chore/*` fra `ai_bot_dev`
- Merge: `ai_bot_dev` → `ai_bot_test` → `ai_trading_pro`

Regler:
- Ingen direkte commits til `ai_bot_test` eller `ai_trading_pro`.
- `ai_bot_test` bruges til soak/integration og regression.
- `ai_trading_pro` er release-branch (stabil).

## 6) PR-processen
En PR skal:
- Linke issue (fx “Closes #57” eller “Relates to #57”).
- Indeholde test evidence (kommando + resultat).
- Respektere DoD (`docs/Definition_of_Done.md`).
- Holde scope small (helst < ~400 linjer diff, medmindre refaktor er planlagt).

Anbefalet:
- 1 review før merge til `ai_bot_dev`.
- Required checks på `ai_bot_dev` (CI).

## 7) CI minimum bar
Minimum på alle PRs:
- Lint/format (ruff/black eller tilsvarende)
- `pytest -q`
- (Valgfrit) `pytest -m integration` på schedule eller manuelt trigger

Hvis noget kræver secrets/keys:
- Brug `-m integration` og skip i default CI.
- Dokumentér tydeligt hvordan tests køres lokalt.

## 8) Release (Beta)
En Beta “release candidate” kræver:
- Gates opfyldt for aktuelle sprint (Sprint 0–N).
- Soak-test på `ai_bot_test` (runtime 24/7 i målperiode).
- Runbooks opdateret for incident + data quality + exchange downtime.

## 9) Incident og drift (kort)
Ved incident:
1) Stop/kill switch (sikkerhed først)
2) Indsaml logs + run-id + metrics
3) Åbn issue: type:bug med repro steps + impact
4) Mitigation først, derefter root cause og regression test
5) Opdater runbook hvis nødvendig
