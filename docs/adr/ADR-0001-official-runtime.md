# ADR-0001: Official runtime

## Status
Accepted

## Kontekst
Repoet indeholder flere historiske og konkurrerende startveje, herunder `main.py`, `run.py web`, `live.py` og Docker-start via uvicorn. Det skaber uklarhed om, hvad den officielle runtime faktisk er.

## Beslutning
Den officielle runtime for `ai_trading_core` er:
- Python target: `bot.live_connector.runner:app`
- Lokal startvej: `python run.py web`
- Container-start: `uvicorn bot.live_connector.runner:app`

## Konsekvenser
- README, Docker og runner-værktøjer skal pege samme sted.
- `bot.engine:app` er ikke officiel runtime.
- `main.py`, `pipeline/core.py` og `run_all.py` behandles som legacy, indtil de enten refaktoreres eller flyttes ud.
- Nye driftsfeatures skal designes omkring live connector-sporet.
