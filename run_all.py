import subprocess
import sys
import glob
import os

PYTHON = sys.executable

# ----- Parametre til pipeline -----
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOOKBACK = 30
FEATURE_VERSION = "v1.0.0"

print(f"🔹 Step 1: Hent rå data fra Binance for {SYMBOL} ({INTERVAL}) ...")
subprocess.run([
    PYTHON, "fetch_data/fetch_binance_data.py",
    "--symbol", SYMBOL,
    "--interval", INTERVAL,
    "--outdir", "data",
    "--lookback", str(LOOKBACK)
], check=True)

# Find nyeste datafil efter fetch
datafiles = sorted(glob.glob(f"data/{SYMBOL}_{INTERVAL}_*.csv"))
if not datafiles:
    print("❌ Ingen datafiler fundet efter fetch. Stopper pipeline.")
    sys.exit(1)
input_file = datafiles[-1]
print(f"✅ Data gemt: {input_file}")

# ----- Step 2: Feature engineering -----
feature_output = f"outputs/feature_data/{SYMBOL.lower()}_{INTERVAL}_features.csv"
print(f"🔹 Step 2: Feature engineering på {input_file} ...")
subprocess.run([
    PYTHON, "features/feature_engineering.py",
    "--input", input_file,
    "--output", feature_output,
    "--version", FEATURE_VERSION
], check=True)

print(f"✅ Features gemt: {feature_output}")

# ----- Step 3: Run engine (model/strategy/main flow) -----
print(f"🔹 Step 3: Run engine ...")
subprocess.run([PYTHON, "bot/engine.py"])
# eller hvis du bruger main.py som controller:
# subprocess.run([PYTHON, "main.py"])

print("🎉 Hele pipeline kørt færdig uden fejl!")
