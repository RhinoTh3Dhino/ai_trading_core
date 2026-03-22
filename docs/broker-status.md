# Broker status

Repoet indeholder to paper-broker spor:

- `bot/brokers/paper.py`
- `bot/brokers/paper_broker.py`

## Midlertidig status

- Begge spor behandles som midlertidige dubletter.
- Nye ændringer i broker-laget bør undgås, indtil der er valgt én kanonisk implementation.
- Runtime-spor bør dokumentere, hvilken broker de bruger.

## Næste beslutning

Der skal gennemføres en separat funktions- og API-sammenligning med fokus på:

- ordre-model
- PnL-model
- logformat
- daily-loss limit
- long/short-understøttelse
- kompatibilitet med GUI og paper runtime
