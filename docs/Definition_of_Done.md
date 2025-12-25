# Definition of Done (DoD)

Dette dokument er "source of truth" for, hvornår en Story/Epic/PR kan betragtes som færdig.

## 1) Story DoD (type:story)

En story er DONE når ALLE punkter er opfyldt:

### Funktionelt
- [ ] Acceptance Criteria (AC) i issuet er opfyldt.
- [ ] Ingen breaking changes uden dokumenteret migration/kompatibilitet.

### Kvalitet & test
- [ ] Relevante unit tests er tilføjet/opdateret.
- [ ] `pytest -q` er kørt lokalt og grønt (eller via CI).
- [ ] Hvis ændringen påvirker integration: marker med `integration` og dokumentér hvordan det køres.

### Drift & observability
- [ ] Structured logging er tilføjet/opdateret hvis ny runtime logik (run-id/correlation hvor relevant).
- [ ] Fejl håndteres tydeligt (fail-fast på config/secrets, informative exceptions).
- [ ] Eventuelle nye config-keys er dokumenteret (README eller relevant docs).

### Dokumentation
- [ ] Hvis ændringen påvirker bruger-/driftflow: docs/runbook opdateret.
- [ ] Hvis ændringen påvirker gates: opdater `docs/SPRINTS_0_6.md`.

### Sikkerhed & compliance
- [ ] Ingen secrets committed.
- [ ] Input valideres hvor relevant (API/config).

---

## 2) Epic DoD (type:epic)

En epic er DONE når:
- [ ] Alle linked stories er CLOSED som DONE (eller eksplicit deferred med begrundelse).
- [ ] Sprint Go/No-Go gates for epics sprint er opfyldt (se `docs/SPRINTS_0_6.md`).
- [ ] Eventuelle kendte risici/teknisk gæld er dokumenteret i epics afsluttende kommentar.

---

## 3) PR DoD

En PR er klar til merge når:
- [ ] PR linker til mindst ét issue (fx “Closes #123”).
- [ ] CI er grøn (minimum: lint + tests).
- [ ] PR-beskrivelse indeholder: hvad/hvorfor/hvordan + test evidence.
- [ ] Relevante docs er opdateret (hvis nødvendigt).
- [ ] Reviewer kan reproducere (kommandoer/steps er tydelige).

---

## 4) Test Evidence (minimum)

Angiv i PR’en:
- Kommando(er) kørt (fx `pytest -q`)
- Resultat (kort output eller “CI grøn: workflow link”)
- Hvis noget ikke kan køres i CI (fx kræver keys): forklar tydeligt og giv alternativ (mock/testnet-markeret integration test).

---

## 5) Definition of Ready (DoR) (hurtig)

En story er READY når:
- [ ] AC/DoD er tydeligt i issuet (eller refererer til relevant doc).
- [ ] Scope er begrænset (1–3 filer eller tydelig modulafgrænsning).
- [ ] Dependencies er kendte (hvis blocked: label `status:blocked` eller kommentar).
