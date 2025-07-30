# tests/test_gridsearch_sanity.py
"""
Unittest for gridsearch_train.py og sanity-pipeline – altid via run.py!
Automatiseret sanity check:
- Pipeline skal eksekvere uden fejl
- Resultater (accuracy, winrate) må ikke være under random baseline
- Eksporterede y_test og y_preds-filer skal matche og kunne læses

Kør: python -m tests.test_gridsearch_sanity
"""

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import subprocess
import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "BTCUSDT_1h_features.csv"
RUN_PATH = PROJECT_ROOT / "run.py"
GRIDSEARCH_REL_PATH = "scripts/gridsearch_train.py"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
Y_TEST_PATH = OUTPUTS_DIR / "y_test.csv"
Y_PREDS_PATH = OUTPUTS_DIR / "y_preds.csv"

def test_gridsearch_pipeline():
    # 0. Slet evt. gamle outputs for at sikre clean test
    for f in [Y_TEST_PATH, Y_PREDS_PATH]:
        if f.exists():
            f.unlink()

    print("[TEST] Starter gridsearch_train.py sanity pipeline via run.py...")
    print(f"[TEST] Kommando: {sys.executable} {RUN_PATH} {GRIDSEARCH_REL_PATH} --input {DATA_PATH} --features close,rsi_14,ema_9 --target target_regime_adapt --balance undersample --test_size 0.4")

    cmd = [
        sys.executable,
        str(RUN_PATH),
        GRIDSEARCH_REL_PATH,
        "--input", str(DATA_PATH),
        "--features", "close,rsi_14,ema_9",
        "--target", "target_regime_adapt",
        "--balance", "undersample",
        "--test_size", "0.4"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print("=== STDOUT ===\n", result.stdout)
    print("=== STDERR ===\n", result.stderr)

    assert result.returncode == 0, f"gridsearch_train.py fejlede ved eksekvering!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert Y_TEST_PATH.exists() and Y_PREDS_PATH.exists(), "Sanity check-filer blev ikke gemt!"

    # 2. Tjek at y_test/y_preds har samme længde og korrekte værdier (kun 0 og 1)
    y_test = pd.read_csv(Y_TEST_PATH, header=None).iloc[:, 0].astype(int).values
    y_preds = pd.read_csv(Y_PREDS_PATH, header=None).iloc[:, 0].astype(int).values
    assert len(y_test) == len(y_preds), f"y_test og y_preds har forskellig længde! y_test: {len(y_test)}, y_preds: {len(y_preds)}"
    assert set(np.unique(y_test)).issubset({0, 1}), f"y_test indeholder ugyldige værdier: {set(np.unique(y_test))}"
    assert set(np.unique(y_preds)).issubset({0, 1}), f"y_preds indeholder ugyldige værdier: {set(np.unique(y_preds))}"

    # 3. Random baseline – check at model er bedre end tilfældighed (50% accuracy)
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, y_preds)
    random_preds = np.random.choice([0, 1], size=len(y_test))
    random_acc = accuracy_score(y_test, random_preds)
    print(f"[TEST] Model accuracy: {acc:.3f}, random baseline: {random_acc:.3f}")
    print("[TEST] Model classification report:\n", classification_report(y_test, y_preds, zero_division=0))
    assert acc >= random_acc - 0.02, f"Model performer dårligere end random baseline! Model: {acc:.3f}, random: {random_acc:.3f}"

    # 4. Ekstra – check at minimum én af klasserne forudsiges
    pred_dist = pd.Series(y_preds).value_counts()
    print("[TEST] Prediction distribution:\n", pred_dist)
    assert (pred_dist > 0).all(), f"Modelen forudsiger kun én klasse! {pred_dist}"

    print("[OK] gridsearch_train.py sanity pipeline bestået.")

if __name__ == "__main__":
    test_gridsearch_pipeline()