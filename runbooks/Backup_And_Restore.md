4. Trading-core stack (bots, DB, MLflow, GUI)

Trading-stack ligger i en anden compose-fil (fx ops/compose/trading/docker-compose.yml eller tilsvarende).
Brug samme princip:

build/pull images

docker compose ... up -d ...

rollback via Git + stable image/config.

(Udbyg når den konkrete compose-fil er fastlagt.)


---

## `runbooks/Backup_And_Restore.md`

*(ingen Linux-line-continuation, men vi sikrer PowerShell-kompatible eksempler)*

```markdown
# Backup & Restore – AI_TRADING_CORE

## 1. Formål

Sikre, at vi kan gendanne:

- TimescaleDB/Postgres-data
- MLflow-artifakter
- Konfiguration, promts, runbooks og docs

med minimal downtime.

---

## 2. Kritisk vs. rebuildable data

Kritisk:

- DB (trades, equity, TE, aggregerede data)
- MLflow-modeller og eksperimenter
- `config/`, `prompts/`, `runbooks/`, `docs/`
- `reports/incidents/`

Rebuildable:

- dele af rå markedsdata, hvis de kan hentes igen fra venues.

---

## 3. Backup-strategi (eksempel)

### 3.1 TimescaleDB/Postgres

Fra en container eller host med `pg_dump`:

```powershell
pg_dump -h timescaledb -U postgres trading | gzip > backups\timescaledb\trading_$(Get-Date -Format yyyy-MM-dd).sql.gz


Tilpas sti og syntaks efter hvordan du kører pg_dump (PowerShell vs. bash).

3.2 MLflow (filsystem)
robocopy C:\mlflow\artifacts C:\backups\mlflow\%DATE% /MIR


(Eller rsync hvis du kører Linux/WSL).

3.3 Config/docs

Alt i Git – sørg for hyppige pushes til remote.

4. Restore (høj-niveau)
4.1 Restore DB

Stop services der bruger DB (bot, evt. feed hvis direkte koblet).

Gendan:

gunzip -c backups\timescaledb\trading_YYYY-MM-DD.sql.gz | psql -h timescaledb -U postgres trading


Start services igen (bot, feed, osv.).

Verificér dashboards og data.

4.2 Restore MLflow

Kopiér artifacts tilbage til MLflow-artifacts dir.

Verificér i UI at eksperimenter og runs er synlige.

4.3 Restore config/docs

git checkout <stable-commit> -- config/ prompts/ runbooks/ docs/

5. Backup-monitorering

Backup-jobs skal logge succes/fejl.

Alarmer, hvis backup ikke køres som planlagt.
