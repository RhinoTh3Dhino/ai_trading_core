# Projektstandarder (AI trading bot)
- Sprog: Python 3.10+
- Test: pytest -q (mål ≥70% dækning på prioriterede moduler)
- Stil: ruff, black, mypy
- Repo: bot/, features/, models/, utils/telegram_utils.py, api/, tests/
- Arbejdsgang: skriv tests først, minimal diff (<40 linjer/fil), targeted pytest
- Sikkerhed: ingen netværkskald; .env / secrets/** må ikke læses
- Claude: brug `plan` som default; skift kun til `acceptEdits` ved små patches (1–3 filer)
