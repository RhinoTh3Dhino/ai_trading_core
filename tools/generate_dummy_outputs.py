import json
import os
import time

import numpy as np

from utils.artifacts import symlink_latest, write_json, write_text

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/backtests", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/feature_data", exist_ok=True)
os.makedirs("outputs/labels", exist_ok=True)

# 1) Metrics (flere filer for at teste rotation)
for i in range(7):
    p = write_json(
        {"sharpe": 1.0 + i * 0.1, "run": i},
        "outputs/metrics",
        "strategy_metrics_btc_1h",
        "v2",
        with_time=False,
    )
    time.sleep(0.2)

# 2) Backtest & chart (dummy)
open("outputs/backtests/backtest_btc_1h_v2_20250101.csv", "w").write("date,pnl\n2025-01-01,10\n")
open("outputs/charts/btc_1h_balance_v2_20250101.png", "wb").write(b"\x89PNG\r\n\x1a\n")

# 3) Models (to stk. for at teste latest)
open("outputs/models/lstm_btc_1h_v2_20250101_1100.keras", "wb").write(b"modelA")
open("outputs/models/lstm_btc_1h_v2_20250102_1200.keras", "wb").write(b"modelB")
symlink_latest(
    "outputs/models/lstm_btc_1h_v2_20250102_1200.keras",
    "outputs/models/best_model.keras",
)

# 4) Feature/labels (dummy)
open("outputs/feature_data/btc_1h_features_v1.0_20250101.csv", "w").write("f1,f2\n1,2\n")
np.save("outputs/labels/btc_1h_labels_v1.0_20250101.npy", np.array([0, 1, 0, 1]))
print("✅ Dummy outputs generated")
