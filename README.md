# AI Trading Core

Dette projekt er fundamentet for en avanceret, robust og modulær AI trading bot bygget til professionelle krav. Den anvender ensemble-strategier, avanceret feature engineering, auto-evaluering, CI/CD, versionering, Telegram-integration og er klar til både personlig brug og SaaS/multi-user udrulning.

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

## 🚀 Quickstart

- **1. Klon repo og installer dependencies:**
- git clone <dit-repo-url>
- cd ai_trading_core
- pip install -r requirements.txt

- **2. Opsæt din .env med TELEGRAM_TOKEN og TELEGRAM_CHAT_ID**
- (se eksempel i .env.example)
  
- **3. Kør botten:**
- python main.py
- Eller én kørsel til test/CI:
- CI=true python main.py

- **3. Tjek resultater i:**
- BotStatus.md – status og performance
- outputs/performance_history.csv – historik og trend
- backups/ – auto-backup af kritiske filer
- Telegram – status/grafer (ved aktiv integration)

---


## 🖥️ CLI-guide & workflows

- **Kør trading-cyklus og Telegram-rapportering:**
- python main.py
  
- **Kør backtest eller retrain:**
- python bot/engine.py --backtest
- python bot/engine.py --train

- **Planlagte jobs kører automatisk via schedule i main.py:**
- Trading-cyklus: hver time
- Daglig status: hver dag kl 08:00
- Retrain: hver dag kl 03:00
- Heartbeat: hver time kl xx:30

---

## 🔁 Pipeline/dataflow

- **1. Data → 2. Features → 3. Modellering → 4. Backtest → 5. Evaluering/rapport → 6. Status/Telegram/trend-graf**
-  flowchart LR
-  A[Hent Data] --> B[Feature Engineering]
-  B --> C[AI/ML Model]
-  C --> D[Backtest & Evaluering]
-  D --> E[Performance-metrics]
-  E --> F[BotStatus.md & Telegram]
-  E --> G[performance_history.csv & graf]

---

## 📈 Output & auto-rapportering

- BotStatus.md: Automatisk opdatering af status og metrics (efter hvert run).

- performance_history.csv: Performance- og balance-historik over tid.

- Auto-backup: Backup af alle centrale filer, roteret dagligt.

- CHANGELOG.md: Opdateres ved hver kørsel.

- Telegram:

   - Automatisk status og heartbeat

   - /status-kommando giver aktuel performance (tekst/graf)

   - Automatisk daglig/ugentlig status, inkl. trend-graf og trade journal

---

## 📊 Historik & trends

- Performance og winrate logges for hvert run.

- Trend-graf genereres automatisk (se outputs/balance_trend.png) og sendes til Telegram.

- Eksempel på auto-genereret trend-graf:

---

## 📬 Telegram-integration

- Status, grafer og trade journal sendes automatisk.

- /status-kommando i Telegram-bot svarer med aktuel performance og graf.

- Planlagt rapportering via schedule/cron, f.eks. daglig kl. 08.

- Robust fejlhåndtering – alle fejl logges og sendes til Telegram ved behov.

---

## 🛠️ Konfiguration og environment

- .env med TELEGRAM_TOKEN og TELEGRAM_CHAT_ID kræves for Telegram-integration.

- Andre hyperparametre og thresholds kan tilpasses i config.json eller direkte i koden.

---

## ❓ FAQ

- Q: Får ModuleNotFoundError: No module named 'utils'?
- A: Kør fra projektroden (cd ai_trading_core). Sørg for at alle mapper har en __init__.py.

- Q: Telegram virker ikke?
- A: Tjek .env for rigtige credentials. Brug evt. test i utils/telegram_utils.py.

- Q: Performance-history eller graf mangler data?
- A: Sørg for, at alle run gennemføres, og at log_performance_to_history() er aktiveret i main.py.

- Q: Hvordan ændrer jeg hvor ofte status/graf sendes?
- A: Redigér tidsplanen i main.py (schedule.every().day.at("08:00")... osv.)

- Q: Hvordan tuner jeg strategi og thresholds?
- A: Brug Optuna-tuning eller justér direkte i config.json.

- Højere threshold = mere selektive signaler, lavere risiko.

- Lavere threshold = flere trades, højere risiko/potentiale.

---

## 🎛️ Tuning & tips

- Ensemble weights & thresholds: Brug load_best_ensemble_params() – så du altid kører med nyeste bedste model.

- Telegram debugging: Sæt DEBUG=true i .env for at se tokens og chat_id.

- Auto-backup: Justér hvor mange dage/kopier du vil gemme i main.py.

- Test-mode: Sæt Telegram-token/chat-id til dummy/test for at teste uden risiko.

---








## 🚀 Funktioner & Arkitektur

- **Automatiseret pipeline:** Fra rå data til Telegram – hele flowet styres med scripts og/eller controller.
- **Avanceret ensemble-voting:** Kombinerer ML, RSI, MACD m.fl. med vægtet voting (Optuna-tuning).
- **Strategi-score & auto-evaluering:** Win-rate, profit, drawdown, Sharpe, Calmar og trades logges og visualiseres.
- **Snapshot/versionering:** Best weights & thresholds gemmes, alle runs loader de nyeste bedste parametre.
- **Feature engineering:** Understøtter mange indikatorer (ATR, EMA/SMA, MACD, RSI, Bollinger Bands m.m.), nemt at tilføje nye.
- **CI/CD + auto-backup:** Automatisk test, backup og changelog-versionering på hver commit.
- **Telegram-integration:** Status, performance, grafer og advarsler sendes løbende (robust fejlhåndtering og heartbeat).
- **Daglig status, auto-retrain og alerting:** Bot rapporterer automatisk status, heartbeat og retrainer ved behov.
- **Robust fejlhåndtering:** Alle trin logger fejl, status og kritiske events til både fil og Telegram.
- **Multi-coin & SaaS-ready:** Bygget til nem udvidelse med flere coins og multi-user/Cloud/SaaS-setup.

---

## 📈 Outputfiler og CSV/Excel-format

**Alle resultater fra walkforward, analyse og top-5/top-10 splits eksporteres automatisk til:**
- `outputs/walkforward/walkforward_summary_<timestamp>.csv/xlsx/json` – Samtlige splits med ALLE nøgletal
- `outputs/walkforward/walkforward_summary_<timestamp>_top5_splits.csv/xlsx/json` – Top-5 bedste splits
- `outputs/walkforward/walkforward_plot_<symbol>_<tf>_<timestamp>.png` – Performance-grafer

**Backup af alle eksportfiler findes i**  
`outputs/walkforward/backup/`

### Felt- og kolonneoversigt (CSV/Excel/JSON)

| Feltnavn                        | Beskrivelse                                                  |
|----------------------------------|-------------------------------------------------------------|
| symbol, timeframe                | Fx BTCUSDT, 1h                                              |
| window_start, window_end         | Split-indeks (relativ til datasættet)                       |
| strategy                        | Anvendt strategi (fx voting_ensemble)                       |
| window_size                      | Antal datapunkter i split                                   |
| train_buyhold_pct / test_buyhold_pct   | Buy & Hold afkast, pct. for split (træning/test)      |
| train_sharpe / test_sharpe       | Sharpe-ratio (annualiseret, train/test)                     |
| train_calmar / test_calmar       | Calmar-ratio (train/test)                                   |
| train_volatility / test_volatility| Volatilitet (annualiseret, train/test)                     |
| train_max_drawdown / test_max_drawdown | Max drawdown (train/test)                         |
| train_win_rate / test_win_rate   | Win-rate i pct. (train/test)                                |
| train_profit_factor / test_profit_factor | Profit factor (train/test)                       |
| train_kelly_criterion / test_kelly_criterion | Kelly-metric (train/test)                     |
| train_expectancy / test_expectancy     | Forventet profit per trade (train/test)             |
| train_total_trades / test_total_trades | Antal handler (train/test)                        |
| train_best_trade / test_best_trade     | Største vinder (train/test, pct.)                  |
| train_worst_trade / test_worst_trade   | Største tab (train/test, pct.)                     |
| train_rolling_sharpe / test_rolling_sharpe | Rullende Sharpe (seneste vindue, train/test) |
| train_trade_duration / test_trade_duration | Gns. varighed af trades i timer                |
| train_regime_drawdown / test_regime_drawdown | Dict/tekst med drawdown pr. regime           |
| ... og evt. regime_drawdown_bull, bear, neutral, osv. (hvis regimes bruges)        |

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

