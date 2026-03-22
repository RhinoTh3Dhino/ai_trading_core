# Cleanup scope v2

Denne cleanup er en repo-saneringsrunde, ikke en fuld funktionel rewrite.

## Omfattet
- Fastlåse officiel runtime
- Gøre installationsveje entydige
- Etablere requirements-profiler
- Dokumentere runtime-spor og broker-dubletter
- Etablere tydelig grænse mellem aktivt og legacy kode

## Ikke omfattet i denne runde
- Fuld omskrivning af bot/engine.py
- Endelig broker-konsolidering mellem paper.py og paper_broker.py
- Fuld flytning af alle legacy-mapper til ny struktur
- Fuld testdækning for hele repoet
- ML-træningspipeline refaktor
