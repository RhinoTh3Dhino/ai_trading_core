import subprocess
import sys
import glob
import os
import argparse

PYTHON = sys.executable

# CLI-parametrisering: så du kan kalde pipelinen fra terminal/GitHub Actions
parser = argparse.ArgumentParser()
parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol (fx BTCUSDT)")
parser.add_argument("--interval", type=str, default="1h", help="Tidsinterval (fx 1h, 4h)")
parser.add_argument("--lookback", type=int, default=30, help="Antal dage at hente data for")
parser.add_argument("--rolling_window", type=int, default=None, help="Antal seneste bars til retraining (valgfri)")
parser.add_argument("--feature_version", type=str, default="v1.0.0", help="Feature-version tag")
args = parser.parse_args()

SYMBOL = args.symbol
INTERVAL = args.interval
LOOKBACK = args.lookback
ROLLING_WINDOW = args.rolling_window
FEATURE_VERSION = args.feature_version

print(f"🔹 Step 1: Hent rå data fra Binance for {SYMBOL} ({INTERVAL}) ...")
fetch_cmd = [
    PYTHON, "fetch_data/fetch_binance_data.py",
    "--symbol", SYMBOL,
    "--interval", INTERVAL,
    "--outdir", "data",
    "--lookback", str(LOOKBACK)
]
if ROLLING_WINDOW:
    fetch_cmd += ["--rolling_window", str(ROLLING_WINDOW)]

subprocess.run(fetch_cmd, check=True)

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

print("🎉 Hele pipeline kørt færdig uden fejl!")
