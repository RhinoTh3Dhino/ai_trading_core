# Runbook — AI Trading Core (Windows)

Denne runbook er en **drift-drejebog** for at køre, overvåge, fejlsøge og gendanne *ai\_trading\_core* lokalt på Windows.

---

## 1) Overblik

* **Repo**: `C:\Users\reno_\Desktop\ai_trading_core`
* **Hovedkomponenter**: `bot.engine`, `utils/*`, `alerts/*`, `core/*`
* **Miljøer**: Lokal Analyze / Paper (simuleret). *(Live er ikke aktiveret i denne runbook)*
* **Artefakter**: `logs/`, `graphs/`, `outputs/`, `models/`
* **Ejerskab**: Reno (primær).

---

## 2) Første-gangs opsætning

1. Åbn PowerShell i repo-roden.
2. (Valgfrit) Opret og aktiver venv:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Installer afhængigheder:

   ```powershell
   pip install -r requirements.txt
   ```
4. Opret `.env` i repo-roden (se næste afsnit).
5. Kør tests for at validere miljøet:

   ```powershell
   pytest -q
   ```

---

## 3) Miljøvariabler (.env)

Opret filen `.env` i repo-roden med mindst:

```ini
# Log & fælles stier
LOG_DIR=logs

# Telegram (krævet for at sende beskeder)
TELEGRAM_TOKEN=<din_bot_token>
TELEGRAM_CHAT_ID=<dit_chat_id>
TELEGRAM_VERBOSE=1              # 0 for stille terminal

# Gating / støjreduktion (valgfrit — sikre defaults findes)
TELEGRAM_DEDUPE_TTL_SEC=120
TELEGRAM_COOLDOWN_GLOBAL_SEC=5
TELEGRAM_COOLDOWN_PER_SYMBOL_SEC=15
TELEGRAM_BATCH_LOWPRIO_EVERY_SEC=60
TELEGRAM_BATCH_MAX_ITEMS=20
TELEGRAM_LOG_DECISIONS=1
```

**Bemærk:** Hvis du vil køre helt uden Telegram (testmode), så udelad eller sæt `TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID` til `dummy_token`/`dummy_id`. Funktionen `telegram_enabled()` vil så returnere *false*, og sendeforsøg bliver logget men ikke sendt.

**Hurtig test uden at røre .env:**

```powershell
$env:TELEGRAM_TOKEN="wrong_token"
$env:TELEGRAM_CHAT_ID="123456789"
python -c "from utils.telegram_utils import send_message; send_message('ping')"
```

---

## 4) Start / Stop

### Analyze (engangs-kørsel, beregner, plotter og sender status)

```powershell
python -m bot.engine --mode analyze --symbol BTCUSDT --interval 1h
```

### Paper (bar-for-bar papirhandel med CSV-artefakter)

```powershell
python -m bot.engine --mode paper --symbol BTCUSDT --interval 1h --alloc-pct 0.10
```

### Stop

* `Ctrl + C` i terminalen.

---

## 5) Health checks (kører det?)

* **Heartbeat til Telegram** (manuel):

  ```powershell
  python -c "from utils.telegram_utils import send_telegram_heartbeat; send_telegram_heartbeat()"
  ```
* **Seneste loglinjer**:

  ```powershell
  Get-Content .\logs\bot.log -Tail 50
  Get-Content .\logs\telegram_log.txt -Tail 50 -Encoding UTF8
  ```
* **Grafer**: `graphs\performance_ml.png`, `graphs\performance_dl.png`, `graphs\performance_ensemble.png`, `graphs\model_comparison.png`
* **CSV-artefakter (paper)**: `logs\fills.csv`, `logs\equity.csv`, `logs\daily_metrics.csv`, `logs\signals.csv`

---

## 6) Planlagte jobs (Windows Task Scheduler)

* **Rotation/vedligehold**: Opgaven `TradeOps-RotateLogs` er konfigureret.
* **Tjek status**:

  ```powershell
  Get-ScheduledTask -TaskName "TradeOps-RotateLogs" | Get-ScheduledTaskInfo
  # LastTaskResult = 0 og NextRunTime skal være fremtidig
  ```

*(Hvis du mangler opgaven, se internt script til oprettelse – eller opret manuelt i Task Scheduler og peg på dit rotationsscript.)*

---

## 7) Drift: typiske workflows

### 7.1 Daglig check

1. Åbn PowerShell i repo-roden
2. Kør analyze engangskørsel og skim output
3. Verificér logs og evt. Telegram heartbeat

### 7.2 Kør papirhandel et interval

1. Start `--mode paper`
2. Hold øje med `logs/` og grafer
3. Stop med `Ctrl + C`

### 7.3 Quick backtest sanity (via tests)

```powershell
pytest -q tests/test_full_pipeline.py::test_full_pipeline_genererer_outputs
```

---

## 8) Fejlsøgning (Troubleshooting)

### 8.1 Telegram

* **404 Not Found** i logs:

  * Forkert `TELEGRAM_TOKEN` eller `TELEGRAM_CHAT_ID` → ret i `.env` og test med `send_message('ping')`.
* **Parse error på Markdown**:

  * Håndteres automatisk: modulet falder tilbage til HTML `<pre>` og sender igen.
* **For mange beskeder / spam**:

  * Sæt `TELEGRAM_VERBOSE=0` for roligere terminal.
  * Skru op for `TELEGRAM_COOLDOWN_*` eller `TELEGRAM_DEDUPE_TTL_SEC`.

### 8.2 Modeller og features

* Manglende modeller: sikre at `models\best_pytorch_model.pt` findes.
* Features: engine vælger auto seneste CSV i `outputs\feature_data\<SYMBOL>_<TF>_latest.csv`. Kør feature-pipeline hvis tom.

### 8.3 Performance-plot mangler

* Tjek at `graphs/` kan skrives til.
* Se evt. exceptions i `bot.log`.

### 8.4 Diskplads / rotation

* Ryd op i `logs/` og `outputs/` eller øg plads.
* Bekræft at `TradeOps-RotateLogs` kører og returnerer `LastTaskResult = 0`.

---

## 9) Tests & coverage

### 9.1 Hele suiten

```powershell
pytest -q
```

### 9.2 Fokuseret Telegram-coverage (nå >60%)

```powershell
pytest -q -o addopts="" `
  tests/test_telegram_utils.py `
  tests/test_telegram_utils_extra.py `
  tests/test_alerts_stack.py::test_telegram_utils_chunk_and_fallback `
  --cov=utils.telegram_utils `
  --cov-report=term-missing `
  --cov-fail-under=60
```

*(Seneste kørsel landede på \~62.5% for `utils.telegram_utils.py`.)*

### 9.3 Hurtige hjælpekommandoer

* Se coverage HTML: `htmlcov/index.html`
* JUnit XML: `reports/junit.xml`
* Coverage XML: `reports/coverage.xml`

---

## 10) Backup & restore

* Backups roteres via tests/backup-utilities (se `tests/backup*.py`).
* **Restore**: Kopiér seneste backup-mappe tilbage til den relevante sti (`outputs/` eller `logs/`).

---

## 11) Sikkerhed & nøgler

* Opbevar `.env` sikkert (ligger uden for VCS).
* Del aldrig `TELEGRAM_TOKEN` offentligt.

---

## 12) Kendte fejl & hurtige fixes

* **`Telegram API 404`** → Ret token/chat\_id; test med `send_message('ping')`.
* **`Can't parse entities`** → automatisk fallback; ingen handling.
* **`Missing file specification after redirection operator` i PowerShell** → det skyldes Bash-heredoc syntaks. Brug PowerShell-venlige eksempler i denne runbook.

---

## 13) Deploy / rollback (lokalt)

1. `git pull`
2. `pytest -q` skal være grønt
3. Start `analyze` eller `paper` som ovenfor
4. Verificér logs og Telegram
5. **Rollback**: `git checkout <forrige-commit>` og gentag trin 2–4

---

## 14) Kontakt & ejerskab

* **Primær**: Reno
* **Sekundær**: (tilføj ved behov)

---

## 15) Bilag — nyttige PowerShell snippets

```powershell
# Tail logs (med Unicode):
Get-Content .\logs\telegram_log.txt -Tail 50 -Encoding UTF8

# Hurtig Telegram ping uden at ændre .env:
$env:TELEGRAM_TOKEN="wrong_token"; $env:TELEGRAM_CHAT_ID="123456789"
python -c "from utils.telegram_utils import send_message; send_message('ping via test')"

# Tjek planlagt opgave:
Get-ScheduledTask -TaskName "TradeOps-RotateLogs" | Get-ScheduledTaskInfo
```
