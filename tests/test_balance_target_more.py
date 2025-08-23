import pandas as pd
import numpy as np
import sys
from pathlib import Path

def test_balance_target_accepts_X_and_target_keyword():
    from features.balance_target import balance_target
    n = 100
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "target": np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)],
    })
    X_bal, y_bal = balance_target(X=df, target="target", method="oversample", random_state=0)
    # Forvent cirka lige fordeling
    vc = y_bal.value_counts()
    assert abs(vc.loc[0] - vc.loc[1]) <= max(1, 0.05 * len(y_bal))

def test_balance_target_cli_smoke(tmp_path, monkeypatch):
    # Dækker main() / CLI-path
    from features import balance_target as bt
    n = 60
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.r_[np.zeros(45, dtype=int), np.ones(15, dtype=int)],
    })
    inp = tmp_path / "in.csv"
    outp = tmp_path / "out.csv"
    df.to_csv(inp, index=False)

    argv = [
        "prog",
        "--input", str(inp),
        "--target", "target",
        "--method", "oversample",
        "--output", str(outp),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bt.main()  # skal ikke crashe og skal skrive fil

    assert outp.exists()
    df_out = pd.read_csv(outp)
    vc = df_out["target"].value_counts()
    assert abs(vc.loc[0] - vc.loc[1]) <= max(1, 0.1 * len(df_out))
