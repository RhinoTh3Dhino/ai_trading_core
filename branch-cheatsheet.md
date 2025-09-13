# Branch Cheatsheet

Denne guide beskriver **hvordan du arbejder med branches** i dit repo med miljø-grenene:

* `ai_bot_dev` (daglig udvikling)
* `ai_bot_test` (release candidate / staging)
* `ai_bot_pro` (prod)

Samt arbejdsgrene:

* `feat/<navn>` (features)
* `fix/<navn>` eller `hotfix/<navn>` (bugfixes)
* `chore/<navn>` (vedligehold, opsætning)
* `release/<version>` (valgfrit – hvis du laver formelle release-branches)

> Alle kommandoer er vist som PowerShell-venlige. Er du på cmd/bash, er de stort set identiske.

---

## 1) Roller og naming

| Gren          | Formål                                               | Kilde                                  | Merge ind i             |
| ------------- | ---------------------------------------------------- | -------------------------------------- | ----------------------- |
| `ai_bot_dev`  | Seneste udviklingsstatus                             | N/A (langliving)                       | `ai_bot_test`           |
| `ai_bot_test` | “Release candidate” til manuel/automatisk validering | Merge fra `ai_bot_dev`                 | `ai_bot_pro`            |
| `ai_bot_pro`  | Produktionskode                                      | Merge fra `ai_bot_test`                | N/A                     |
| `feat/...`    | Afgrænset feature                                    | oprettes fra `ai_bot_dev`              | `ai_bot_dev`            |
| `hotfix/...`  | Kritiske rettelser                                   | fra `ai_bot_pro` (eller `ai_bot_test`) | `ai_bot_pro` + backport |
| `chore/...`   | Opsætning, CI, refactor                              | fra `ai_bot_dev`                       | `ai_bot_dev`            |

**Navngivningseksempler**

* `feat/streaming-mvp`
* `feat/observability-metrics`
* `hotfix/telegram-rate-limit`
* `chore/ci-pin-pydantic`

---

## 2) Daglig rytme (hurtig opskrift)

```powershell
# 0) Hent alt og vær up-to-date
git fetch --all --prune

# 1) Start en ny feature fra dev
git switch ai_bot_dev
git pull --ff-only
git switch -c feat/streaming-mvp   # vælg dit navn

# 2) Arbejd, commit småt og ofte
git add -A
git commit -m "feat(streaming): live connector quiet logs + metrics lag"

# 3) Skub første gang (opret remote-branch)
git push -u origin feat/streaming-mvp

# 4) Hold din feature ajour (rebase på dev før PR)
git fetch origin
git rebase origin/ai_bot_dev
# løs konflikter -> `git add ...` -> `git rebase --continue`
git push --force-with-lease

# 5) Lav PR -> merge til ai_bot_dev når tests er grønne
# (enten via GitHub UI eller CLI)
```

---

## 3) Hvornår laver du en ny feature-gren?

* Når opgaven er **mere end et par commits** eller **kan bryde** noget (f.eks. “Streaming features (MVP)”, “Observability /metrics”).
* Når du vil **arbejde isoleret** fra dev uden at blokere andre ting.
* Når du vil have **en ren PR-diff** og CI-historik for lige den ændring.

---

## 4) Merge-kriterier (dev → test → pro)

* **Dev**: Merge når:

  * Alle tests/CI ✅
  * Lokal smoke fungerer
  * PR-beskrivelse forklarer ændringen kort
* **Test**: Merge fra `ai_bot_dev` når:

  * Feature-sæt til release er samlet
  * Evt. manuel test er ✅
* **Pro**: Merge fra `ai_bot_test` når:

  * Alt er stabilt i test
  * Versions-tag oprettet (se Release-afsnit)

---

## 5) PR-tjekliste

* [ ] Meningsfuld titel: `feat: streaming MVP – quiet logs + parquet schema`
* [ ] Beskrivelse: Hvad/ hvorfor / evt. migration
* [ ] Tests kører grønt
* [ ] Ingen støj: fjern debug/kommenterede linjer
* [ ] Opdater docs/cheatsheets hvis relevant

---

## 6) Synk din feature med `ai_bot_dev` (rebase-flow)

```powershell
git fetch origin
git switch feat/streaming-mvp
git rebase origin/ai_bot_dev
# Løs konflikter -> git add ... -> git rebase --continue
git push --force-with-lease
```

**Hvorfor rebase?** Ren historik og en PR der kun viser dine ændringer oven på seneste dev.

**Protip**

```powershell
# Husk sikker varianten – beskytter mod at overskrive andres arbejde
git push --force-with-lease
```

---

## 7) Merge til `ai_bot_dev`

Når PR er klar og grøn:

* **Squash & merge** (oftest bedst): Samler dine commits til én ryddelig ændring.
* **Merge commit** (hvis du vil bevare commit-granularitet).

---

## 8) Hotfix-flow (prod-kritisk)

```powershell
# Start hotfix direkte fra pro
git switch ai_bot_pro
git pull --ff-only
git switch -c hotfix/telegram-rate-limit

# Fix -> commit -> push
git add -A
git commit -m "fix(telegram): handle rate limit gracefully"
git push -u origin hotfix/telegram-rate-limit

# PR til ai_bot_pro (hurtig)
# Når merged: backport til test og dev
git switch ai_bot_test
git pull --ff-only
git merge --no-ff hotfix/telegram-rate-limit   # eller cherry-pick -x <sha>
git push

git switch ai_bot_dev
git pull --ff-only
git merge --no-ff hotfix/telegram-rate-limit
git push
```

**Alternativ backport** (enkelt commit):

```powershell
git switch ai_bot_test
git cherry-pick -x <SHA-fra-pro>
git push
```

---

## 9) Release-flow (tags og version)

Simpelt skema via tags (uden release-branch):

```powershell
# Når test → pro er godkendt
git switch ai_bot_pro
git pull --ff-only

# Versionér og tag
$VERSION="v1.3.0"
git tag -a $VERSION -m "Release $VERSION: streaming MVP + metrics"
git push origin $VERSION
```

**Tips**

```powershell
# Brug beskrivende version i build scripts
git describe --tags --always
```

---

## 10) Konflikter & afbrud

```powershell
# Under rebase/merge konfliktløsning
git status
# ... ret filer ...
git add <fil>
git rebase --continue   # eller: git merge --continue

# Fortryd rebase hvis det stak af
git rebase --abort

# Parkér lokale ændringer hurtigt
git stash -u           # gem alt (inkl. untracked)
git stash pop          # hent tilbage
```

---

## 11) Oprydning

```powershell
# Slet lokal branch efter merge
git branch -d feat/streaming-mvp

# Slet remote branch
git push origin --delete feat/streaming-mvp

# Fjern gamle remote referencer
git fetch --all --prune
```

---

## 12) Alias & kvalitet af liv (valgfrit)

`~/.gitconfig`:

```ini
[alias]
  lg = log --graph --decorate --oneline --all
  st = status -sb
  co = checkout
  sw = switch
  rb = rebase
  pl = pull --ff-only
  ps = push
[rerere]
  enabled = true
```

**`rerere.enabled=true`** lærer Git at huske dine konfliktløsninger til næste gang.

---

## 13) Beskyttelse af branches (GitHub)

* Protect `ai_bot_dev`, `ai_bot_test`, `ai_bot_pro`

  * Kræv grøn CI før merge
  * Kræv PR (ingen direkte push til pro/test)
  * Begræns “force push” (tillad kun på feature branches)

---

## 14) Typiske scenarier fra projektet

### A) “Streaming features (MVP)”

```powershell
git switch ai_bot_dev
git pull --ff-only
git switch -c feat/streaming-mvp
# ... arbejde, commits ...
git push -u origin feat/streaming-mvp

# opdatér løbende:
git fetch
git rebase origin/ai_bot_dev
git push --force-with-lease

# PR → merge til ai_bot_dev
```

### B) “Observability (/metrics)”

```powershell
git switch ai_bot_dev
git pull --ff-only
git switch -c feat/observability-metrics
# ... arbejde ...
git push -u origin feat/observability-metrics
# rebase løbende, PR når klar
```

### C) Hurtig CI-fix (chore)

```powershell
git switch ai_bot_dev
git pull --ff-only
git switch -c chore/ci-pin-pydantic
# ret requirements/ci.yml
git commit -m "chore(ci): pin pydantic to fix schemas import in CI"
git push -u origin chore/ci-pin-pydantic
# PR → merge til ai_bot_dev
```

---

## 15) Fejlsøgning – de klassiske

**“pathspec ‘feat/…’ did not match …”**
→ Branch findes ikke lokalt endnu (eller du er ikke på korrekt directory).

```powershell
git fetch --all --prune
git branch -a
git switch -c feat/navn   # eller: git switch feat/navn hvis den findes remote
```

**“cannot rebase: You have unstaged changes.”**
→ Gem eller commit inden rebase.

```powershell
git add -A && git commit -m "WIP"   # eller
git stash -u
git rebase origin/ai_bot_dev
git stash pop
```

**“DID NOT RAISE CancelledError” i tests**
→ Async task stoppede selv – cancel ikke nødvendigt. Tilpas test til ikke at forvente exception efter `task.cancel()` hvis din kode allerede håndterer shutdown.

---

## 16) Commit-konvention (kort)

* `feat(scope): beskrivelse`
* `fix(scope): beskrivelse`
* `chore(scope): beskrivelse`
* `docs(scope): …`
* `test(scope): …`
* `refactor(scope): …`

Eksempel: `feat(streaming): quiet logs + lag metrics window`

---

## 17) Hurtig tjekliste før merge til **pro**

* [ ] `ai_bot_pro` er up-to-date
* [ ] `ai_bot_test` → `ai_bot_pro` PR godkendt
* [ ] Tag oprettet og skubbet: `git tag -a vX.Y.Z && git push origin vX.Y.Z`
* [ ] CI grøn og noter skrevet (CHANGELOG.md)

---

**Det var det.** Denne opskrift holder dig effektiv selv som solo-udvikler: du får ren historik, sikre promotion-trin (`dev → test → pro`) og hurtig backport ved hotfixes – og du undgår branch-snavs.
