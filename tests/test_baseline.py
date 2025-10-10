# tests/test_baseline.py
from __future__ import annotations

import re
from pathlib import Path

# Headless backend (virker i CI og uden display)
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Robust PROJECT_ROOT:
# 1) Prøv utils.project_path.PROJECT_ROOT
# 2) Fald tilbage til repo-roden (to niveauer op fra denne fil)
try:
    from utils.project_path import PROJECT_ROOT as _PRJ

    PROJECT_ROOT = Path(_PRJ).resolve()
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stier vi bruger i testen
DATA_CSV = PROJECT_ROOT / "data" / "BTCUSDT_1h_with_target.csv"
OUT_DIR = PROJECT_ROOT / "outputs"


def _pick_target_column(columns: list[str]) -> str | None:
    """
    Vælg 'target' hvis til stede; ellers første kolonne der matcher ^target(\b|_).
    Returnerer None hvis ingen passende kolonne findes.
    """
    if "target" in columns:
        return "target"
    for c in columns:
        if re.match(r"^target(\b|_)", c):
            return c
    return None


@pytest.mark.timeout(20)
def test_baseline_smoke(tmp_path: Path | None = None) -> None:
    """
    Baseline smoke-test:
    - Hvis data findes: læs CSV, find en target-kolonne (generisk 'target' eller første 'target_*'),
      lav et simpelt plot og gem det.
    - Hvis data IKKE findes: skip testen pænt (CI må ikke fejle på manglende lokale artefakter).
    - Rapporter (via prints) om evt. ekstra artefakter, men uden at fejle.
    """
    if not DATA_CSV.exists():
        pytest.skip(
            f"Mangler datafil: {DATA_CSV}. "
            "Skip i CI – generér lokalt via dit data-script, hvis du vil køre testen fuldt."
        )

    import pandas as pd

    df = pd.read_csv(DATA_CSV)

    # Vælg target-kolonne robust
    target_col = _pick_target_column(list(df.columns))
    if target_col is None:
        pytest.skip(
            "Ingen kolonner der matcher 'target' eller 'target_*' i CSV – "
            "springer testen over i stedet for at fejle."
        )

    # Gør target numerisk for en sikker rolling-mean (håndter bool/str/NaN)
    tgt = pd.to_numeric(df[target_col], errors="coerce")

    # Plot (rolling mean) og gem headless
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / f"{target_col}_rolling_mean.png"
    plt.figure(figsize=(8, 3))
    tgt.rolling(100).mean().plot(title=f"{target_col} (rolling mean)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    assert (
        plot_path.exists() and plot_path.stat().st_size > 0
    ), "Plot blev ikke gemt korrekt"

    # Ikke-kritisk info: feature-importance billede (kun log)
    fi_img = OUT_DIR / "feature_importance_baseline.png"
    if fi_img.exists():
        print(f"[INFO] Feature importance fundet: {fi_img}")
    else:
        print("[INFO] Feature importance-billede ikke fundet (ok i CI).")

    # Ekstra info til logs
    try:
        vc = df[target_col].value_counts(normalize=True, dropna=False)
        print("[INFO] Target fordeling (normaliseret):")
        print(vc.to_string())
    except Exception as e:
        print(f"[INFO] Kunne ikke udskrive value_counts for '{target_col}': {e}")
