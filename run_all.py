import subprocess
import sys

# Brug sys.executable for at sikre du bruger samme python-miljø/venv!
PYTHON = sys.executable

# 1. Hent ny data
subprocess.run([PYTHON, "fetch_data/fetch_binance_data.py"])

# 2. Feature engineering
subprocess.run([PYTHON, "features/feature_engineering.py"])

# 3. Run engine/main
subprocess.run([PYTHON, "bot/engine.py"])
# eller:
# subprocess.run([PYTHON, "main.py"])
