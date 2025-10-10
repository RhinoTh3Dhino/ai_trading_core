from pathlib import Path

import numpy as np
import pandas as pd

n = 2000
ts = pd.date_range("2023-01-01", periods=n, freq="1h")
price = 20000 + np.cumsum(np.random.randn(n)) * 10
high = price + np.random.rand(n) * 5
low = price - np.random.rand(n) * 5
open_ = price + np.random.randn(n)
vol = np.random.rand(n) * 100

df = pd.DataFrame(
    {
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": vol,
        "ema_9": pd.Series(price).ewm(span=9, adjust=False).mean(),
        "ema_21": pd.Series(price).ewm(span=21, adjust=False).mean(),
        "rsi_14": np.clip(50 + np.random.randn(n) * 5, 0, 100),
    }
)

outdir = Path("outputs/feature_data")
outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "BTCUSDT_1h_latest.csv", index=False)
print("Wrote", outdir / "BTCUSDT_1h_latest.csv")
