# tests/test_baseline.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Brug headless backend i CI / uden display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Robust PROJECT_ROOT:
# 1) Prøv at importere fra utils.project_path
# 2) Ellers fald tilbage til mappen to niveauer op (repo-roden)
try:
    from utils.project_path import PROJECT_ROOT as _PRJ
    PROJECT_ROOT = Path(_PRJ).resolve()
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stier vi bruger i testen
DATA_CSV = PROJECT_ROOT / "data" / "BTCUSDT_1h_with_target.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
PLOT_PATH = OUT_DIR / "target_rolling_mean.png"
FI_IMG = OUT_DIR / "feature_importance_baseline.png"


@pytest.mark.timeout(20)
def test_baseline_smoke(tmp_path: Path | None = None) -> None:
    """
    Baseline smoke-test:
    - Hvis data findes: læs CSV, lav et simpelt plot og gem det.
    - Hvis data IKKE findes: skip testen pænt (CI må ikke fejle på manglende lokale artefakter).
    - Rapporter (via prints) om evt. ekstra artefakter, men uden at fejle.
    """
    if not DATA_CSV.exists():
        pytest.skip(
            f"Mangler datafil: {DATA_CSV}. "
            "Skip i CI – generér lokalt via dit data-script, hvis du vil køre testen fuldt."
        )

    # --- 1) Læs data og lav en let kontrol ---
    import pandas as pd

    df = pd.read_csv(DATA_CSV)

    # Kræv at kolonnen eksisterer, ellers er der noget galt med datasættet
    assert "target" in df.columns, "CSV mangler kolonnen 'target'"

    # --- 2) Plot (rolling mean) og gem headless ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    # Tåler numerisk/boolean target – brug mean() direkte
    df["target"].rolling(100).mean().plot(title="Target (TP-hit) over tid")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    # Plot skal være skrevet
    assert PLOT_PATH.exists() and PLOT_PATH.stat().st_size > 0, "Plot blev ikke gemt korrekt"

    # --- 3) Ikke-kritisk: feature-importance billede (kun info) ---
    if FI_IMG.exists():
        print(f"[INFO] Feature importance fundet: {FI_IMG}")
    else:
        print("[INFO] Feature importance-billede ikke fundet (ok i CI).")

    # Ekstra info til logs
    vc = df["target"].value_counts(normalize=True, dropna=False)
    print("[INFO] Target fordeling (normaliseret):")
    print(vc.to_string())
