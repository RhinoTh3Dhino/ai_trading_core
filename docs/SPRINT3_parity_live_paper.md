# Sprint 3 – Paritet + live/paper-integration (dev-PC)

## Formål

Sprint 3 sikrer, at vi har en **robust bro** mellem:

- Backtest (offline, på historiske data)
- Paper-trading (simuleret live, men uden rigtige ordrer)

Fokus er BTCUSDT 1h og Flagship Trend v1.

---

## 1. Standard-paritet: backtest vs. paper

### 1.1 Standard CLI’er

**Backtest (standard baseline)**

```bash
python -m scripts.run_backtest_standard \
    --symbol BTCUSDT \
    --interval 1h \
    --features auto \
    --no-persist

Paper (standard baseline)

python -m scripts.run_paper_standard \
    --symbol BTCUSDT \
    --interval 1h \
    --features auto \
    --tag v1

Forudsætning – rå data

python -m scripts.fetch_raw_ohlcv_binance \
    --symbol BTCUSDT \
    --interval 1h \
    --limit 2000


Standard-flowet forventer, at AUTO-feature pipelinen har skrevet:

outputs/data/features_auto/BTCUSDT_1h_features.csv

1.2 Paritetstest

Testfil: tests/backtest/test_flagship_trend_v1_parity.py

Relevante tests:

test_flagship_trend_v1_backtest_vs_paper_parity

Kørsel (uden global coverage):

pytest -q -o addopts="" \
    tests/backtest/test_flagship_trend_v1_parity.py::test_flagship_trend_v1_backtest_vs_paper_parity


Testen:

Kører fetch_raw_ohlcv_binance

Kører run_backtest_standard

Kører run_paper_standard

Sammenligner metrics-filer:

outputs/backtests/btcusdt_1h_v1.json

outputs/paper/btcusdt_1h_v1.json

Kriterier (current state):

num_trades og profit_pct skal matche inden for defineret tolerance.

2. Flagship-paritet: Flagship backtest vs. Flagship paper
2.1 CLI’er

Flagship backtest

python -m scripts.run_backtest_flagship_v1 \
    --symbol BTCUSDT \
    --interval 1h \
    --tag dev1 \
    --no-persist


Output:

outputs/backtests/flagship_btcusdt_1h_dev1_trades.csv

outputs/backtests/flagship_btcusdt_1h_dev1_equity.csv

outputs/backtests/flagship_btcusdt_1h_dev1.json

Flagship paper

python -m scripts.run_paper_flagship_v1 \
    --symbol BTCUSDT \
    --interval 1h \
    --tag dev1


Output:

outputs/paper/flagship_btcusdt_1h_dev1_trades.csv

outputs/paper/flagship_btcusdt_1h_dev1_equity.csv

outputs/paper/flagship_btcusdt_1h_dev1.json

Begge bruger samme AUTO-features:

outputs/data/features_auto/BTCUSDT_1h_features.csv

2.2 Flagship-paritetstest (loose)

Testfil: tests/backtest/test_flagship_trend_v1_parity.py

Test:

test_flagship_trend_v1_flagship_backtest_vs_paper_parity_loose

Kørsel:

pytest -q -o addopts="" \
    tests/backtest/test_flagship_trend_v1_parity.py::test_flagship_trend_v1_flagship_backtest_vs_paper_parity_loose


Testen:

Henter rå data (Binance → outputs/data/BTCUSDT_1h_raw.csv).

Kører run_backtest_flagship_v1 med tag dev1.

Kører run_paper_flagship_v1 med samme symbol/interval/tag.

Læser:

outputs/backtests/flagship_btcusdt_1h_dev1.json

outputs/paper/flagship_btcusdt_1h_dev1.json

Sammenligner bl.a.:

num_trades (identisk)

profit_pct (inden for løs tolerance)

Samme fortegn på profit_pct (dvs. begge vinder/taber)

Det giver en smoke-test, der sikrer, at Flagship-backtest og Flagship-paper ikke divergerer voldsomt på samme datasæt.

3. Udvikler-flow på dev-PC (live/paper bridge)

Standard “fra nul til paritet” på udviklings-PC:

# 1) Hent rå data
python -m scripts.fetch_raw_ohlcv_binance \
    --symbol BTCUSDT \
    --interval 1h \
    --limit 2000

# 2) (Hvis nødvendigt) byg AUTO-features
#   - via eksisterende feature-pipeline
#   - forventet fil: outputs/data/features_auto/BTCUSDT_1h_features.csv

# 3) Kør Flagship backtest
python -m scripts.run_backtest_flagship_v1 \
    --symbol BTCUSDT \
    --interval 1h \
    --tag dev1 \
    --no-persist

# 4) Kør Flagship paper
python -m scripts.run_paper_flagship_v1 \
    --symbol BTCUSDT \
    --interval 1h \
    --tag dev1

# 5) (Valgfrit) kør paritetstest
pytest -q -o addopts="" \
    tests/backtest/test_flagship_trend_v1_parity.py::test_flagship_trend_v1_flagship_backtest_vs_paper_parity_loose


Efter kørsel har vi:

Backtest:

outputs/backtests/flagship_btcusdt_1h_dev1.json

Paper:

outputs/paper/flagship_btcusdt_1h_dev1.json

De to filer bruges som reference-punkt for videre kalibrering af FillEngine/paper-adapter.

4. Kendte begrænsninger (Sprint 3)

Paritet er pt. “loose”:

Vi kræver ikke perfektion på alle metrics.

Fokus er: samme antal trades, samme retning på profit og fornuftigt niveau.

ML/DL-pipelinen for Flagship kan køre i fallback-mode (random / simple regler),
så længe vi primært bruger dette setup til struktur- og paritetstest.

Denne dokumentation dækker kun dev-PC/live-agtig paper-trading.
Migrering til homelab/docker/grafana håndteres i senere epics.


Det er nok til at C3.4 har en tydelig skriftlig reference.

---

## 3. README / CLI-overblik (kort snippet)

Tilføj evt. dette i `README.md` under en sektion “Sprint 3 – Paritet & Paper CLI”:

```markdown
### Sprint 3 – Paritet & paper CLI (dev-PC)

Standard baseline:

```bash
# Backtest
python -m scripts.run_backtest_standard --symbol BTCUSDT --interval 1h --features auto --no-persist

# Paper
python -m scripts.run_paper_standard --symbol BTCUSDT --interval 1h --features auto --tag v1


Flagship Trend v1:

# Flagship backtest
python -m scripts.run_backtest_flagship_v1 --symbol BTCUSDT --interval 1h --tag dev1 --no-persist

# Flagship paper
python -m scripts.run_paper_flagship_v1 --symbol BTCUSDT --interval 1h --tag dev1


Paritetstests:

# Standard paritet
pytest -q -o addopts="" tests/backtest/test_flagship_trend_v1_parity.py::test_flagship_trend_v1_backtest_vs_paper_parity

# Flagship paritet (loose)
pytest -q -o addopts="" tests/backtest/test_flagship_trend_v1_parity.py::test_flagship_trend_v1_flagship_backtest_vs_paper_parity_loose


---

## 4. Oprydning – konkrete punkter

Når du lægger ovenstående docs ind, vil jeg anbefale:

1. **Emoji/UTF-8 oprydning i CLI-scripts**
   Vi har allerede ramt `UnicodeEncodeError` pga. `cp1252`.
   - Sikr at alle `print()` i scripts under `scripts/` og `engine.py` bruger **ASCII-only**:
     - Fx `"AUTO features valgt -> ..."` i stedet for emojis.
   - Du har allerede rettet nogle – lav en hurtig søgning på `🧩`, `📈` osv. og fjern dem i CLI-scripts.

2. **Docstrings-opdatering**
   - I `scripts/run_paper_standard.py`, `scripts/run_backtest_flagship_v1.py`, `scripts/run_paper_flagship_v1.py`:
     - Sørg for, at top-docstring forklarer kort:
       - Hvad scriptet gør.
       - Hvilke filer det læser/skriver.
       - Hvordan det relaterer til Sprint 3 / Flagship.

3. **Pytest marker-dokumentation (kort)**
   - Hvis du har en `tests/README.md` eller lignende, tilføj en note om:
     - `@pytest.mark.heavy` → bruges til Binance-hit / E2E-paritets-tests.
     - Eksempel på, hvordan du kun kører de tunge paritetstests, når du vil.

---

## 5. Definition of Done for C3.4

Du kan markere C3.4 som *done*, når:

- [ ] `docs/SPRINT3_parity_live_paper.md` findes i repoet med indhold ca. som ovenfor.
- [ ] README har en kort CLI-oversigt for Sprint 3 (standard + Flagship + tests).
- [ ] Emojis/UTF-8-problemer er fjernet fra relevante CLI-scripts, så Windows-konsollen ikke fejler.
- [ ] Testene:
  - `test_flagship_trend_v1_backtest_vs_paper_parity`
  - `test_flagship_trend_v1_flagship_backtest_vs_paper_parity_loose`

  stadig kører grønt med:
  ```bash
  pytest -q -o addopts="" tests/backtest/test_flagship_trend_v1_parity.py
