# Cleanup scope v1

Denne cleanup er en **repo-saneringsrunde**, ikke en fuld funktionel rewrite.

## Omfattet

- Fastlåse officiel runtime
- Gøre installationsveje entydige
- Rydde op i requirements-profiler
- Gøre `run.py` konsistent med Docker og README
- Dokumentere runtime-spor og broker-dubletter
- Etablere tydelig grænse mellem aktivt og legacy kode

## Ikke omfattet i denne runde

- Fuld omskrivning af `bot/engine.py`
- Endelig broker-konsolidering mellem `paper.py` og `paper_broker.py`
- Flytning af alle legacy-mapper til ny struktur
- Fuld testdækning for hele repoet
- ML-træningspipeline refaktor

## Kriterier for succes

1. En ny udvikler kan starte den officielle service uden at gætte.
2. Repoet har én dokumenteret runtime-sandhed.
3. Dependency-profilerne er forståelige og adskilte.
4. Legacy-spor er dokumenteret som legacy.
