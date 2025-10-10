# scripts/seed_demo_data.py
"""
Generér syntetiske data til GUI/API-demo:
- logs/equity.csv
- logs/daily_metrics.csv
- logs/fills.csv
- api/sim_signals.json
Kør: python scripts/seed_demo_data.py
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
API_DIR = ROOT / "api"
LOGS.mkdir(parents=True, exist_ok=True)
API_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# --- Equity (60 dage) ---
n_days = 60
start_equity = 100_000.0
dates = [date.today() - timedelta(days=n_days - 1 - i) for i in range(n_days)]
daily_ret = np.random.normal(
    loc=0.0004, scale=0.006, size=n_days
)  # syntetisk drift/vol
equity = [start_equity]
for r in daily_ret[1:]:
    equity.append(equity[-1] * (1 + r))
equity = np.array(equity)
peak = np.maximum.accumulate(equity)
drawdown_pct = (equity - peak) / peak * 100.0

df_eq = pd.DataFrame(
    {
        "date": [d.isoformat() for d in dates],
        "equity": np.round(equity, 2),
        "cash": np.round(equity, 2),
        "positions_value": np.zeros(n_days, dtype=float),
        "drawdown_pct": np.round(drawdown_pct, 2),
    }
)
(df_eq).to_csv(LOGS / "equity.csv", index=False)

# --- Daglige metrikker (30 dage) ---
rows = []
for i in range(30):
    d = date.today() - timedelta(days=29 - i)
    signal_count = random.randint(12, 36)
    trades = random.randint(6, signal_count)
    win_rate = round(random.uniform(48.0, 61.0), 2)
    gross_pnl = round(random.uniform(50, 350), 2)
    net_pnl = round(gross_pnl - random.uniform(5, 25), 2)
    max_dd = round(random.uniform(-6.0, -1.0), 2)
    sharpe_d = round(random.uniform(0.4, 1.3), 2)
    rows.append(
        [
            d.isoformat(),
            signal_count,
            trades,
            win_rate,
            gross_pnl,
            net_pnl,
            max_dd,
            sharpe_d,
        ]
    )

pd.DataFrame(
    rows,
    columns=[
        "date",
        "signal_count",
        "trades",
        "win_rate",
        "gross_pnl",
        "net_pnl",
        "max_dd",
        "sharpe_d",
    ],
).to_csv(LOGS / "daily_metrics.csv", index=False)

# --- Fills (100 handler) ---
fills = []
for k in range(100):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    side = "BUY" if k % 2 == 0 else "SELL"
    qty = round(random.uniform(0.01, 0.2), 4)
    price = round(50_000 + random.uniform(-1_500, 1_500), 2)
    commission = round(price * qty * 0.0002, 4)
    pnl_realized = round(random.uniform(-12, 18), 2)
    fills.append([ts, "BTCUSDT", side, qty, price, commission, pnl_realized])

pd.DataFrame(
    fills,
    columns=["ts", "symbol", "side", "qty", "price", "commission", "pnl_realized"],
).to_csv(LOGS / "fills.csv", index=False)

# --- Mock-signaler til API/GUI ---
now = int(datetime.utcnow().timestamp())
signals = []
for k in range(40):
    signals.append(
        {
            "timestamp": now - k * 60,
            "symbol": "BTCUSDT",
            "side": "BUY" if k % 2 == 0 else "SELL",
            "confidence": round(random.uniform(0.52, 0.82), 2),
            "price": round(50_000 + random.uniform(-800, 800), 2),
            "sl": 0.985,
            "tp": 1.015,
            "regime": random.choice(["trend", "range"]),
        }
    )
signals.reverse()
(API_DIR / "sim_signals.json").write_text(
    json.dumps(signals, indent=2), encoding="utf-8"
)

print("✅ Seedet: logs/ + api/sim_signals.json")
