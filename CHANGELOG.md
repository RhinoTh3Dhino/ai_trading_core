## [0.1.0] – 2025-06-07

**Seneste ændringer:**

- dc9e90d RhinoTh3Dhino 2025-06-07 feat: Step 3 – Telegram-rapport, daglig status og hjertelyd
- b71b5fc RhinoTh3Dhino 2025-06-05 feat: Implementeret automatisk backtest og profitberegning (Sprint 2 – Step 2)
- 238f338 RhinoTh3Dhino 2025-06-05 feat: Implementeret fuldautomatisk model-evaluering, logging og visualisering (Sprint 2 – Step 1)
- 5a418ed RhinoTh3Dhino 2025-06-02 Tilføjet og rettet fuldautomatisk pipeline: - Automatisk omdøbning af 'datetime' til 'timestamp' i engine.py for robust backtest - Korrekt ML-feature-handling: kun model-features til predict, fuld df til backtest - Forbedret model_training.py: returnerer feature_cols, robust til DataFrame eller CSV - Forbedret backtest.py: fejlsikring af kolonner, klar fejlhåndtering og debug - Grafgenerering til balance og drawdown, med automatisk navngivning og backup - Telegram-integration med både tekst og billeder (status, balance, drawdown) - Alle scripts, moduler og mapper opdateret og testet - Koden er nu CI/CD-klar og matcher masterplanens Step 3
- e48e09a RhinoTh3Dhino 2025-06-02 test: Tilføjet robust test med dummy-data og verificeret model-log og modelgem
- 42ec30a RhinoTh3Dhino 2025-05-31 chore: Opdateret .gitignore, model-træning, unittest og eval-logging
- 9409db9 RhinoTh3Dhino 2025-05-31 feat: Fuldt automatiseret modeltræning, evaluering og logging – pipeline fra features til metrics CSV, klar til næste sprint
- 888eafa RhinoTh3Dhino 2025-05-31 fix: Robust feature engineering – håndterer separator og datatyper, klar til næste step
- d574509 RhinoTh3Dhino 2025-05-31 feat: Tilføjet robust validering af CSV-data og opdateret data-workflow
- fa93357 RhinoTh3Dhino 2025-05-31 test: Tilføjet og rettet unittest for hent_binance_data – mock Binance OHLCV, robust filtest

---

## [v0.4.0] - 2025-06-07

- Tilføjet robust_utils.py med safe_run til alle vigtige scripts
- Import-struktur gjort robust via sys.path.append-trick
- Automatisk Telegram-integration med tekst og billeder i hele flowet
- Automatisk fejlhåndtering og alarm via Telegram
- Visualiseringer (balance, drawdown, confusion matrix) gemmes og sendes automatisk
- Versions- og backup-flow valideret og testet
- Klar til cronjobs, batch eller SaaS-deployment


## [Sprint A: Strategi-optimering] - 2025-06-30

### Added
- Binance downloader med 2 års historik og flere timeframes
- Grid search modul til optimering af SL, TP, EMA-parametre
- CSV-log af alle gridsearch-resultater og top-strategier

### Changed
- Opdateret features_pipeline.py (robusthed, alle indikatorer med)
- Forbedret paper_trader.py til batch og journal pr. run
- Performance-metrics udvidet (Sharpe, Sortino, win-rate, profit factor mm.)

### Fixed
- Robust håndtering af edge cases i performance, gridsearch, feature pipeline
- Alle test scripts og pipeline virker med nye data

### Next
- Out-of-sample test af top-5 strategier
- Automatisk strategi-udvælgelse og baseline update
- Forberedelse til ML/AI-signal sprint


## [vX.X.X] – 2025-07-04

### Added
- Automatisk eksport af walkforward-resultater til CSV, XLSX og JSON
- Backup af alle analysefiler, inkl. top-5 og top-10 splits for hurtig videre analyse
- Kolonner til markering af bedste split og top-5 i CSV-output
- Robust håndtering af NaN/inf og udvidet test af eksport/backup-funktioner
- Generering af plots for Sharpe, Winrate/Buy&Hold og Rolling Sharpe på tværs af splits
- README.md udvidet med konkret analyseguide, filbeskrivelser og brugseksempler
- analyze_walkforward.py udvidet med pivottabeller og fleksible plots
- Kvalitetssikring af hele eksport- og analyse-flowet

### Changed
- Telegram-upload nu med alle relevante eksport-filer (CSV, XLSX, JSON)
- Refaktorering for mere robust og modulær kode

### Fixed
- Diverse edge-cases og fejl i eksport/backup/analyse-håndtering

## [vX.Y.Z] – 2025-07-07

### Added
- Fuldt modulært analyze_strategies.py script med CLI-argumenter.
- Automatisk logging af strategi-performance (ML, RSI, MACD, Ensemble) i outputs/strategy_scores.csv.
- Generering af strategi-score bar chart og historisk heatmap for win-rate.
- Markdown-rapport for hvert run (`outputs/report_<runid>.md`) med metrikker og grafer.
- Automatisk opdatering af BotStatus.md.
- Advarsler/alarmer ved lav win-rate eller høj drawdown.
- Alt output samlet i outputs/-mappen for let backup, CI/CD og analyse.
- Auto-åbning af output-mappen efter run for nem visuel feedback.

### Changed
- Forberedt integration med Telegram og egen backtest-funktion.

## [vX.Y.Z] – 2025-07-07

### Added
- Komplet analyze_features.py med klassisk, permutation og SHAP-feature-importance.
- CLI/argument-support og auto-output til outputs/-mappen.
- Automatisk markdown-rapport, PNG og CSV for importance.
- analyze_feature_pruning.py: Fuld auto-feature-pruning pipeline med test af top-N features, accuracy-plot og markdown-rapport.
- Understøttelse af suppression af joblib/loky-warnings og robust SHAP-integration.
- Alle outputs gemmes og versionsstyres for let analyse, CI/CD og audit trail.

### Changed
- Forbedret pipelines for modularitet, integration og next-level analyse.

### Fixed
- Rettet SHAP-fejl ved brug med RandomForest/XGBoost via TreeExplainer.
- Løst issue med tabulate-dependency for markdown-rapportering.


## [vX.Y.Z] – 2025-07-07

### Added
- Komplet analyze_regimes.py med fuld regime-analyse (bull/bear/neutral) for alle strategier (ML, RSI, MACD, Ensemble).
- Automatisk generering af regime-metrics: win-rate, profit, drawdown og antal handler pr. regime.
- Visualisering af regime-performance (PNG-grafer) for hver strategi.
- Markdown-rapport med samling af grafer og nøgletal for audit og SaaS/Notion.
- Robust struktur for udvidelse til Telegram/Notion-rapportering og adaptiv strategi.

### Changed
- Modularisering og automatisering af regime-flow, så det let kan integreres i pipeline eller CI/CD.

### Fixed
- Ingen kendte fejl – struktur og output valideret på real data.

### 2025-07-07 – step5: Dokumentation & Auto-rapportering færdiggjort

- README.md opdateret: quickstart, CLI, dataflow, tuning, FAQ og eksempler.
- Telegram-status, graf og filafsendelse virker end-to-end.
- Auto BotStatus.md, historik, trend-graf og backup på hvert run.
- Daglig/ugentlig status via schedule/main.py, inkl. heartbeat og retrain.
- Alle tests og edge-cases dækket – 100% grøn test-suite.

## [v1.0.0] - 2025-07-10
### Added
- Fuldt Deep Learning-modul med `train_pytorch.py`:
    - CLI-træning og integration til engine/pipeline
    - Automatisk håndtering af feature-selection, meta-header og numeriske kolonner
    - Class weights for robusthed ved ubalanceret target
    - Logging af træningsforløb, validation accuracy, confusion matrix og classification report
    - Automatisk retrain-loop og Telegram-beskeder
    - Stabil og fejltolerant PyTorch-træning på både GPU/CPU

### Fixed
- Sikret, at pipeline og engine kan bruge/loade nye modeller uden manuelle ændringer



## [2025-05-23]
✅ Step 4 test


## [2025-05-23]
- Step 5: Automatisk changelog test


## [2025-05-23]
- Step 5: Automatisk changelog test

### 2025-05-24 05:07:41
✅ Bot kørte og lavede backup: backups\backup_2025-05-24_05-07-41
---

### 2025-05-24 05:27:55
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_05-27-55
---

### 2025-05-24 12:56:15
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_12-56-15
---

### 2025-05-24 13:05:41
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_13-05-41
---

### 2025-05-24 13:12:46
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_13-12-46
---

### 2025-05-24 13:23:53
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_13-23-53
---

### 2025-05-24 13:37:21
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_13-37-21
---

### 2025-05-24 13:39:21
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_13-39-21
---

### 2025-05-24 13:45:01
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_13-45-01
---

### 2025-05-24 14:07:01
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-07-01
---

### 2025-05-24 14:09:44
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-09-44
---

### 2025-05-24 14:14:51
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-14-51
---

### 2025-05-24 14:19:57
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-19-57
---

### 2025-05-24 14:26:35
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-26-35
---

### 2025-05-24 14:31:11
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-31-11
---

### 2025-05-24 14:36:01
✅ Bot kørte og lavede backup: backups\2025-05-24\backup_14-36-01
---

### 2025-05-26 17:36:49
✅ Bot kørte og lavede backup: backups\2025-05-26\backup_17-36-48
---

### 2025-05-26 19:04:00
✅ Bot kørte og lavede backup: backups\2025-05-26\backup_19-03-59
---

### 2025-05-26 19:24:43
✅ Bot kørte og lavede backup: backups\2025-05-26\backup_19-24-42
---

### 2025-05-29 20:24:50
✅ Bot kørte og lavede backup: backups\2025-05-29\backup_20-24-49
---

### 2025-05-31 00:05:37
✅ Bot kørte og lavede backup: backups\2025-05-31\backup_00-05-36
---

### 2025-05-31 00:48:57
✅ Bot kørte og lavede backup: backups\2025-05-31\backup_00-48-57
---

### 2025-05-31 15:56:47
✅ Bot kørte og lavede backup: backups\2025-05-31\backup_15-56-47
---

### 2025-06-05 15:38:30
✅ Bot kørte og lavede backup: backups\2025-06-05\backup_15-38-30


### 2025-06-07 03:18:55
✅ Bot kørte og lavede backup: backups\2025-06-07\backup_03-18-55
---


## [2025-06-07]
- Step 5: Automatisk changelog test

### 2025-06-09 23:02:53
✅ Bot kørte og lavede backup: backups\2025-06-09\backup_23-02-53
---

### 2025-06-13 20:34:50
✅ Bot kørte og lavede backup: backups\2025-06-13\backup_20-34-50
---

### 2025-06-13 21:29:45
✅ Bot kørte og lavede backup: backups\2025-06-13\backup_21-29-45
---

### 2025-06-14 00:06:46
✅ Bot kørte og lavede backup: backups\2025-06-14\backup_00-06-46
---

### 2025-06-14 00:12:24
✅ Bot kørte og lavede backup: backups\2025-06-14\backup_00-12-24
---

### 2025-06-14 00:18:27
✅ Bot kørte og lavede backup: backups\2025-06-14\backup_00-18-27
---

### 2025-06-14 00:21:35
✅ Bot kørte og lavede backup: backups\2025-06-14\backup_00-21-35
---

### 2025-06-14 02:35:58
✅ Bot kørte og lavede backup: backups\2025-06-14\backup_02-35-58
---

### 2025-06-16 19:42:12
✅ Bot kørte og lavede backup: backups\2025-06-16\backup_19-42-11
---

### 2025-06-16 20:00:05
✅ Bot kørte og lavede backup: backups\2025-06-16\backup_20-00-04
---

### 2025-06-16 21:00:03
✅ Bot kørte og lavede backup: backups\2025-06-16\backup_21-00-03
---


## [2025-06-16]
- Step 5: Automatisk changelog test


## [2025-06-16]
- Step 5: Automatisk changelog test


## [2025-06-17]
- Step 5: Automatisk changelog test


## [2025-06-17]
- Step 5: Automatisk changelog test

### 2025-06-17 21:59:20
✅ Bot kørte og lavede backup: backups\2025-06-17\backup_21-59-20
---

### 2025-06-17 22:00:03
✅ Bot kørte og lavede backup: backups\2025-06-17\backup_22-00-03
---


## [2025-06-18]
- Step 5: Automatisk changelog test


### 2025-06-18 19:37:27
✅ Bot kørte og lavede backup: backups\2025-06-18\backup_19-37-27
---

### 2025-06-18 19:43:30
✅ Bot kørte og lavede backup: backups\2025-06-18\backup_19-43-30
---

### 2025-06-18 20:00:03
✅ Bot kørte og lavede backup: backups\2025-06-18\backup_20-00-03
---

### 2025-06-18 20:05:07
✅ Bot kørte og lavede backup: backups\2025-06-18\backup_20-05-06
---
=======

## [2025-06-18]
- Step 5: Automatisk changelog test



## [2025-06-18]
- Step 5: Automatisk changelog test


### 2025-06-19 00:19:31
❌ Bot fejlede: Cannot cast array data from dtype('O') to dtype('float64') according to the rule 'safe'
---

### 2025-06-19 00:32:47
❌ Bot fejlede: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
---

### 2025-06-20 17:41:08
❌ Bot fejlede: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
---

### 2025-06-20 21:09:53
❌ Bot fejlede: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
---

### 2025-06-20 21:17:56
❌ Bot fejlede: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.
---

### 2025-06-20 21:24:15
❌ Bot fejlede: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.
---

### 2025-06-20 21:29:15
❌ Bot fejlede: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.
---

### 2025-06-20 21:35:22
❌ Bot fejlede: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.
---

### 2025-06-20 21:53:10
❌ Bot fejlede: unhashable type: 'numpy.ndarray'
---

### 2025-06-20 22:02:46
❌ Bot fejlede: unhashable type: 'numpy.ndarray'
---

### 2025-06-20 22:35:20
❌ Bot fejlede: only length-1 arrays can be converted to Python scalars
---

### 2025-06-20 22:43:47
❌ Bot fejlede: only length-1 arrays can be converted to Python scalars
---

### 2025-06-20 22:50:54
❌ Bot fejlede: unhashable type: 'numpy.ndarray'
---

### 2025-06-20 23:00:14
❌ Bot fejlede: unhashable type: 'numpy.ndarray'
---

### 2025-06-20 23:10:19
❌ Bot fejlede: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 2 with shape (15, 3, 3) and arg 3 with shape (15,).
---

### 2025-06-20 23:18:56
❌ Bot fejlede: too many indices for array: array is 0-dimensional, but 1 were indexed
---

### 2025-06-20 23:46:11
❌ Bot fejlede: ufunc 'absolute' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> None
---

### 2025-06-20 23:55:56
❌ Bot fejlede: ufunc 'absolute' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> None
---

### 2025-06-21 00:02:14
❌ Bot fejlede: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- volume_spike

---

### 2025-06-21 00:09:19
✅ Bot kørte og lavede backup: backups\2025-06-21\backup_00-09-18
---

### 2025-06-21 00:39:17
✅ Bot kørte og lavede backup: backups\2025-06-21\backup_00-39-17
---
=======

## [2025-06-18]
- Step 5: Automatisk changelog test



### 2025-06-21 01:28:28
❌ Bot fejlede: 'regime'
---

### 2025-06-21 01:33:50
❌ Bot fejlede: 'regime'
---

### 2025-06-21 01:37:33
❌ Bot fejlede: 'regime'
---

### 2025-06-21 01:44:15
❌ Bot fejlede: 'regime'
---

### 2025-06-21 01:49:22
❌ Bot fejlede: 'regime'
---

### 2025-06-21 01:53:44
❌ Bot fejlede: 'regime'
---

### 2025-06-21 02:02:28
✅ Bot kørte og lavede backup: backups\2025-06-21\backup_02-02-28
---

### 2025-06-21 02:13:15
✅ Bot kørte og lavede backup: backups\2025-06-21\backup_02-13-14
---

### 2025-06-21 02:17:45
✅ Bot kørte og lavede backup: backups\2025-06-21\backup_02-17-45
---

### 2025-06-21 02:33:53
✅ Bot kørte og lavede backup: backups\2025-06-21\backup_02-33-53
---
=======

## [2025-06-20]
- Step 5: Automatisk changelog test



### 2025-06-23 23:07:22
✅ Bot kørte og lavede backup: backups\2025-06-23\backup_23-07-21
---
=======

## [2025-06-23]
- Step 5: Automatisk changelog test


### 2025-06-24 21:45:06
✅ Bot kørte og lavede backup: backups\2025-06-24\backup_21-45-06
---

### 2025-06-24 22:00:09
✅ Bot kørte og lavede backup: backups\2025-06-24\backup_22-00-09
---
=======

## [2025-06-24]
- Step 5: Automatisk changelog test



## [2025-06-26]
- Step 5: Automatisk changelog test


### 2025-06-26 18:42:01
✅ Bot kørte og lavede backup: backups\2025-06-26\backup_18-42-00
---

### 2025-06-26 19:57:50
✅ Bot kørte og lavede backup: backups\2025-06-26\backup_19-57-50
---

### 2025-06-26 20:00:08
✅ Bot kørte og lavede backup: backups\2025-06-26\backup_20-00-08
---

### 2025-06-26 21:27:03
✅ Bot kørte og lavede backup: backups\2025-06-26\backup_21-27-02
---

### 2025-06-26 21:30:01
✅ Bot kørte og lavede backup: backups\2025-06-26\backup_21-30-01
---

### 2025-06-26 22:00:00
✅ Step 1: Automatisk datafetch implementeret i fetch_binance_data.py (CLI, logging, fallback, Telegram)
---



## [2025-06-26]
- Step 5: Automatisk changelog test


### Step 1: Automatisk Datafetch – færdig

- Henter rå OHLCV-data fra Binance via CLI eller pipeline.
- Data gemmes som CSV i korrekt struktur (timestamp, open, high, low, close, volume).
- Robust fejlhåndtering med fallback til seneste fil.
- Logning til både BotStatus.md og Telegram (inkl. emoji/status).
- Kompatibel med Windows og Linux (UTF-8).
- Output-mappe oprettes automatisk.
- Klar til udvidelse med multi-coin/multi-timeframe og videre pipeline-steps.
=======

## [2025-06-26]
- Step 5: Automatisk changelog test



## [2025-06-26]
- Step 5: Automatisk changelog test


## [2025-06-27]
- Step 5: Automatisk changelog test


## [2025-06-27]
- Step 5: Automatisk changelog test


## [2025-06-27]
- Step 5: Automatisk changelog test


## [2025-06-28]
- Step 5: Automatisk changelog test


### 2025-06-28 02:28:57
✅ Bot kørte og lavede backup: backups\2025-06-28\backup_02-28-56
---

### 2025-06-30 17:03:21
✅ Bot kørte og lavede backup: backups\2025-06-30\backup_17-03-20
---
=======

## [2025-06-28]
- Step 5: Automatisk changelog test



## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-06-30]
- Step 5: Automatisk changelog test


## [2025-07-01]
- Step 5: Automatisk changelog test


## [2025-07-01]
- Step 5: Automatisk changelog test


## [2025-07-01]
- Step 5: Automatisk changelog test


## [2025-07-01]
- Step 5: Automatisk changelog test


## [2025-07-04]
- Step 5: Automatisk changelog test


## [2025-07-04]
- Step 5: Automatisk changelog test


## [2025-07-07]
- Step 5: Automatisk changelog test


## [2025-07-07]
- Step 5: Automatisk changelog test


## [2025-07-07]
- Step 5: Automatisk changelog test


## [2025-07-07]
- Step 5: Automatisk changelog test


### 2025-07-07 21:17 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-07 23:03 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-07 23:05 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-07 23:13 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test



### 2025-07-08 00:05 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:12 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:14 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:18 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:21 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:34 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:37 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:42 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 00:47 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 01:02 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 01:07 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-07 21:35 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test


### 2025-07-07 23:15 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test

### 2025-07-07 23:22 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test

### 2025-07-07 23:28 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test


### 2025-07-08 01:52 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-07 23:37 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test



### 2025-07-08 02:11:45
✅ Bot kørte og lavede backup: backups\2025-07-08\backup_02-11-45
---

### 2025-07-08 02:20 - vvTEST - TEST001
- Første changelog-test.

### 2025-07-08 02:20:48
✅ Bot kørte og lavede backup: backups\2025-07-08\backup_02-20-47
---

### 2025-07-07 23:57 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-07]
- Step 5: Automatisk changelog test



### 2025-07-10 11:49:20
✅ Bot kørte og lavede backup: backups\2025-07-10\backup_11-49-19
---

### 2025-07-08 00:28 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-08]
- Step 5: Automatisk changelog test


### 2025-07-10 12:35 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-10]
- Step 5: Automatisk changelog test

### 2025-07-10 13:18 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-10]
- Step 5: Automatisk changelog test

### 2025-07-10 20:13 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-10]
- Step 5: Automatisk changelog test

### 2025-07-10 22:22 - vvTEST - TEST001
- Første changelog-test.


## [2025-07-10]
- Step 5: Automatisk changelog test

### 2025-07-11 23:32 - vvTEST - TEST001
- Første changelog-test.
