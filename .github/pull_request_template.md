## Linked Issue
Closes #___
Relates to #___

Issue-type:
- [ ] type:story
- [ ] type:bug
- [ ] type:chore
- [ ] type:epic (sjældent)

## Hvad ændrer PR’en?
-

## Hvorfor?
-

## Scope (hold den lille)
- Primære filer/moduler:
- Berørte flows (data/backtest/paper/live/ui/api/ops):
- Breaking change?
  - [ ] Nej
  - [ ] Ja (kræver migration-notat nedenfor)

## Konfiguration / Secrets
- Nye config keys?
  - [ ] Nej
  - [ ] Ja (list dem her + opdater docs/.env.example)
- Kræver secrets/keys?
  - [ ] Nej
  - [ ] Ja (forklar hvordan tests køres uden secrets, fx mocks/integration-marker)

## Hvordan testes det?
Kommandoer kørt:
- [ ] `pytest -q`
- [ ] (valgfrit) `pytest -q -m integration`

Test evidence (kort):
- CI-run/commit:
- Lokal output (kort):

## DoD-check (krævet)
- [ ] AC opfyldt (se issue)
- [ ] Tests grønne (lokalt og/eller CI)
- [ ] Logging/fejlhåndtering ok (hvor relevant)
- [ ] Docs opdateret hvis relevant (`docs/Definition_of_Done.md`, runbooks, `docs/SPRINTS_0_6.md`)
- [ ] Ingen secrets i commit

## Go/No-Go gate påvirkning
- Sprint: ___ (fx Sprint 3)
- Epic: #___
- Gate reference: `docs/SPRINTS_0_6.md` (kort: hvilken gate og hvordan den opfyldes)

## Risiko og rollback
- Risiko:
- Rollback-plan:

## Migration-notat (kun hvis breaking change)
- Hvad bryder?
- Hvordan migreres?
- Hvem påvirkes?
