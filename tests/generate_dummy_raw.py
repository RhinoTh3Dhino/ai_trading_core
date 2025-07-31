import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import os
import pandas as pd
import numpy as np

from utils.project_path import PROJECT_ROOT

# AUTO PATH CONVERTED
os.makedirs(PROJECT_ROOT / "outputs" / "data", exist_ok=True)
symbols = ["btcusdt", "ethusdt", "dogeusdt"]
timeframes = ["1h", "4h"]

for symbol in symbols:
    for tf in timeframes:
        n = 500
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=n, freq="H" if tf == "1h" else "4H"
                ),
                "open": np.random.uniform(25000, 35000, n),
                "high": np.random.uniform(25500, 35500, n),
                "low": np.random.uniform(24500, 34500, n),
                "close": np.random.uniform(25000, 35000, n),
                "volume": np.random.uniform(10, 1000, n),
            }
        )
        # AUTO PATH CONVERTED
        path = fPROJECT_ROOT / "outputs" / "data/{symbol}_{tf}_raw.csv"
        df.to_csv(path, index=False)
        print(f"✅ Dummy data gemt: {path}")
