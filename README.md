# AI Trading Core
Dette projekt er fundamentet for en avanceret, robust og modulær AI trading bot bygget til professionelle krav. Den anvender ensemble-strategier, avanceret feature engineering, auto-evaluering, CI/CD, versionering, Telegram-integration og er klar til både personlig brug og SaaS/multi-user udrulning.

# 🚀 Funktioner & Arkitektur (Sprint 3+)
- Automatiseret pipeline: Fra rå data til Telegram – hele flowet styres med scripts og/eller controller.

- Avanceret ensemble-voting: Kombinerer ML, RSI, MACD (flere kan tilføjes) med vægtet voting (Optuna-tuning).

- Strategi-score & auto-evaluering: Win-rate, profit, drawdown og trades logges og visualiseres for ML, RSI, MACD, Ensemble.

- Snapshot/versionering: Best weights & thresholds gemmes, alle runs loader de nyeste bedste parametre.

- Feature engineering: Støtter mange indikatorer (ATR, EMA/SMA, MACD, RSI, Bollinger Bands m.m.), nemt at tilføje nye.

- CI/CD + auto-backup: Automatisk test, backup og changelog-versionering på hver commit.

- Telegram-integration: Status, performance, grafer og advarsler sendes løbende (robust fejlhåndtering og heartbeat).

- Daglig status, auto-retrain og alerting: Bot rapporterer automatisk status, heartbeat, og retrainer ved behov.

- Robust fejlhåndtering: Alle trin logger fejl, status og kritiske events til både fil og Telegram.

- Multi-coin & SaaS-ready: Bygget til nem udvidelse med flere coins og multi-user/Cloud/SaaS-setup.

# 📈 Seneste opdateringer
- Automatiseret pipeline: run_all.py styrer data → features → labels → model → eval → Telegram i ét flow.

- Ny strategi-score: Automatisk pr. strategi, inklusive regime-stats.

- Auto-versionering af features, labels og modeller (meta-data og snapshots).

- CI/CD opdateret: .gitignore blokerer alle store/temp/miljøfiler.

- Telegram-rapportering: Grafer og metrics sendes, inklusive fejl og backup-status.

---

## 🗂️ **Mappestruktur (uddrag)**

ai_trading_core/
│
├── bot/                  # Engine, strategi og telegram scripts
├── features/             # Feature engineering scripts
├── fetch_data/           # Data-fetch og hentning fra Binance
├── models/               # ML-modeller, best_model.pkl, snapshots
├── tuning/               # Optuna, tuner-cache, tuning logs/results
├── outputs/              # Feature-CSV, grafer, evals, backup (ikke i git)
├── data/                 # Eval-filer, eksempelfiler, testdata
├── logs/                 # Run-logs, fejllogs, Telegram logs
├── .github/workflows/    # CI/CD og auto-backup scripts
├── tests/                # Testdata og test-scripts
├── main.py               # Hoved-controller (starter schedule-loop)
├── run_all.py            # Automatisk pipeline fra data til eval
├── requirements.txt      # Alle Python dependencies
└── .gitignore            # Beskytter alle temp/store/miljøfiler




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

