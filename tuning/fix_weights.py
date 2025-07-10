import json

# Filstien til din ensemble-param-fil
param_path = "tuning/best_ensemble_params.json"

# Antal strategier du bruger i ensemblet (fx ML, RSI, MACD, EMA = 4)
N_STRATEGIES = 4

# Standardvægt hvis der mangler weights
DEFAULT_WEIGHT = 1.0

# Load og ret param-fil automatisk
with open(param_path, "r") as f:
    params = json.load(f)

weights = params.get("weights", [])
# Ret længden, hvis mismatch:
if len(weights) < N_STRATEGIES:
    # Tilføj default-vægte hvis der mangler
    weights += [DEFAULT_WEIGHT] * (N_STRATEGIES - len(weights))
elif len(weights) > N_STRATEGIES:
    # Trim listen hvis for mange
    weights = weights[:N_STRATEGIES]

params["weights"] = weights

with open(param_path, "w") as f:
    json.dump(params, f, indent=2)

print(f"✅ Weights opdateret til korrekt længde: {weights}")
