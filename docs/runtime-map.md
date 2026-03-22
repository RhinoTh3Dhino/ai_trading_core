# Runtime map

Denne fil er den operationelle klassifikation af repoets kendte entrypoints og hovedspor.

## Entry points

- `bot.live_connector.runner:app` — ACTIVE — officiel runtime
- `python run.py web` — TARGET — bør pege på officiel runtime
- `Dockerfile` — ACTIVE — officiel container-start
- `bot.engine:app` — EXPERIMENTAL — engine-web
- `live.py` — EXPERIMENTAL — paper/live daemon
- `main.py` — LEGACY — ældre scheduler/runner
- `pipeline/core.py` — LEGACY — ældre analyze/inference-pipeline
- `run_all.py` — LEGACY — ældre orkestrering

## Regler

1. Nye runtime-ændringer må ikke introducere endnu et entrypoint.
2. README, Docker og `run.py web` skal pege på samme officielle runtime.
3. Legacy-spor må ikke markedsføres som standard-flow.
