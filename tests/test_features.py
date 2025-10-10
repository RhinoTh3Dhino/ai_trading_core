import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.project_path import PROJECT_ROOT

# Korrekt tilføjelse af projektrod til sys.path (ALTID som str!)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))

from features.ta_indicators import add_ta_indicators

# ---------- KONFIG ----------
CSV_PATH = Path(PROJECT_ROOT) / "data" / "BTCUSDT_1h.csv"

# ---------- DUMMYDATA HVIS CSV MANGLER ----------
if not CSV_PATH.exists():
    print("⚠️  CSV ikke fundet, opretter dummy testdata...")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 500
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="H"),
            "open": np.random.uniform(25000, 35000, n),
            "high": np.random.uniform(25500, 35500, n),
            "low": np.random.uniform(24500, 34500, n),
            "close": np.random.uniform(25000, 35000, n),
            "volume": np.random.uniform(10, 1000, n),
        }
    )
    # Brug altid decimal="." (det er standard!)
    df.to_csv(str(CSV_PATH), index=False, sep=";", decimal=".")
    print(f"Dummydata gemt som {CSV_PATH}")

# ---------- LOAD OG STANDARDISÉR KOLONNENAVNE ----------
df = pd.read_csv(str(CSV_PATH), sep=";", decimal=".")
df.columns = [c.lower() for c in df.columns]

# ---------- AUTOMATISK KOLONNEMAPNING ----------
if "datetime" in df.columns and "timestamp" not in df.columns:
    df = df.rename(columns={"datetime": "timestamp"})

# ---------- TJEK FOR NØDVENDIGE KOLONNER ----------
expected = {"open", "high", "low", "close", "volume", "timestamp"}
if not expected.issubset(set(df.columns)):
    print("🚨 Mangler én eller flere af følgende kolonner:", expected)
    print("Fandt kolonner:", df.columns)
    sys.exit(1)

# ---------- KONVERTER TIL NUMERIC ----------
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- KONVERTER TIL DATETIME & SÆT SOM INDEX ----------
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# ---------- KØR FEATURES-PIPELINE ----------
features_df = add_ta_indicators(df)

# ---------- PRINT OG TJEK ----------
print(features_df.tail(10))
print("NaN per kolonne:\n", features_df.isna().sum())
print("Inf per kolonne:\n", np.isinf(features_df).sum())

# ---------- PLOT EKSEMPEL ----------
if "ema_21" in features_df.columns and "ema_200" in features_df.columns:
    features_df[["close", "ema_21", "ema_200"]].plot(figsize=(12, 5))
    plt.title("Pris med EMA21 og EMA200")
    plt.show()
else:
    print("En eller flere EMA-kolonner mangler til plot.")
