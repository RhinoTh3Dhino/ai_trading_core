# AI Trading Core

Dette projekt er fundamentet for en avanceret, robust og modulær **AI trading bot**, der benytter **ensemble-strategier**, avanceret feature engineering, automatisk strategi-score og CI/CD workflows – klar til både personlig og kommerciel brug.

---

## 🚀 Funktioner & Arkitektur (v. Sprint 3)

- **Avanceret Ensemble-voting:** Kombinerer ML-model, RSI, MACD (og snart flere) i ét samlet signal med vægtet voting (weights tunet via Optuna).
- **Strategi-score & Evaluering:** Automatisk evaluering og sammenligning af hver strategi (ML, RSI, MACD, Ensemble) med win-rate, profit, drawdown og antal handler.
- **Optuna-tuning af Threshold & Weights:** Automatisk tuning af både ensemble weights og thresholds for optimal performance.
- **Snapshot & Versionering:** Bedste weights og threshold gemmes som versioneret JSON snapshot – alle runs loader automatisk de nyeste, bedste parametre.
- **Feature Engineering:** Understøtter flere tekniske indikatorer (ATR, VWAP, Bollinger Bands, EMA/SMA etc.).
- **CI/CD + Backup:** Fuldt workflow for automatiske tests, backup af data og modeller, og versioneret changelog.
- **Telegram-integration:** Status, performance, grafer og advarsler sendes løbende til Telegram (og robust fejlhåndtering i CI/test).
- **Daglig status, heartbeat og retrain:** Botten rapporterer løbende status, og kan udvides til automatisk retrain ved lav performance.
- **Robust fejlhåndtering og logging:** Alle kritiske funktioner logger fejl og opdaterer BotStatus.md og CHANGELOG.md.

---

## 📈 **Seneste opdateringer (Sprint 3 – Delmål 4, Step 1: Strategi-score & Evaluering)**

- **Ny strategi-score**: Nu får du automatisk beregnet og visualiseret win-rate, profit, drawdown og antal handler **pr. strategi** (ML, RSI, MACD, Ensemble).
- **Alt scores og rapporteres** til både konsol, Telegram og log.
- **Modul metrics.py**: Indeholder alle core-metrics og evaluering på tværs af strategier.
- **Engine pipeline**: Loader altid de bedste weights/thresholds og evaluerer strategi-performance i samme run.

---

## 🗂️ **Mappestruktur (uddrag)**



---

## 🧠 **Standard Commit Guide**

1. Tilføj ændringer  
   `git add .`

2. Commit med beskrivende besked  
   `git commit -m "feat: Tilføjet strategi-score og ensemble-evaluering"`

3. (Ekstra) Tilføj detaljeret changelog i næste linjer.

4. Push til korrekt branch  
   `git push origin ai_bot_dev`

**Tips:**  
- Commit ofte, men meningsfuldt – hver commit skal kunne forklares.
- Brug branches konsekvent: `ai_bot_dev`, `ai_bot_test`, `ai_trading_pro`.
- Husk at merge dev → test → prod, og brug GitHub Actions til auto-tests og backup.
- Opdater CHANGELOG.md løbende (automatisk hvis muligt).

---

## 🔁 **CI/CD Milestones og Merge-flow**

1. Udvikling i `ai_bot_dev` → test i `ai_bot_test` → release i `ai_trading_pro`.
2. Automatisk backup af alle kernefiler og log/metrics for hver run.
3. Daglig status og performance rapporteres til Telegram og BotStatus.md.

---

## 📊 **Testplan og Robusthed**

- Fejlhåndtering for alle kritiske funktioner (backup, status, Telegram, .env)
- Telegram-funktioner mock-testes for robusthed
- Edge-cases for cleanup/backup og retrain
- Automatisk tests via GitHub Actions
- Robust multi-run og parallelle pipelines på tværs af flere coins

---

## 📝 **Changelog – Seneste vigtige ændringer**

- **Strategi-score & evaluering** tilføjet (metrics.py)
- **Automatisk tuning af ensemble weights og thresholds (Optuna)**
- **Snapshot og versionering af bedste parametre**
- **Robust Telegram-integration** (og fejlhåndtering)
- **Auto-backup og CI/CD flows**
- **Multi-strategi pipeline – nemt at udvide med nye indikatorer**

---

## 📌 **Kommende features (roadmap)**

- Udvidelse med flere tekniske indikatorer og strategier
- Visualisering af strategi-score og performance over tid (dashboard)
- Automatisk regime-analyse og “auto-retrain”
- CLI/Telegram-kommandoer til tuning og status
- Automatisk rapportering og analyser til Notion/README

---

## 📚 **Referencer**

- [Notion masterplan og roadmap](#)
- [CHANGELOG.md](CHANGELOG.md)
- [BotStatus.md](BotStatus.md)

---

## Kontakt og bidrag

Har du spørgsmål, idéer eller vil bidrage? Skriv i Issues eller kontakt via Telegram!

---

