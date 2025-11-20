# tests/persistence/conftest.py
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest


@pytest.fixture()
def tmp_layout(monkeypatch, tmp_path: Path):
    """
    Patcher PERSIST-layout til et midlertidigt workspace under tmp_path.
    Sikrer at både config.config og utils.artifacts læser samme PERSIST.
    """
    # Importér først for at få referencerne
    import config.config as cfg
    import utils.artifacts as art

    # Lav en dyb kopi for at undgå sideeffekter på global PERSIST
    persist: Dict = deepcopy(cfg.PERSIST)
    root = tmp_path / "outputs"
    live_root = root / "live"
    persist["LAYOUT"]["ROOT"] = str(root)
    persist["LAYOUT"]["LIVE_ROOT"] = str(live_root)

    # Sæt forudsigelige thresholds så tests er deterministiske
    persist["ROTATE_MAX_ROWS"] = 2
    persist["ROTATE_MAX_MINUTES"] = 0  # tidsbaseret rotation bruges ikke i disse tests

    # Patch begge moduler
    monkeypatch.setattr(cfg, "PERSIST", persist, raising=True)
    monkeypatch.setattr(art, "PERSIST", persist, raising=True)

    # Opret roden
    (tmp_path / "archives").mkdir(parents=True, exist_ok=True)
    Path(persist["LAYOUT"]["LIVE_ROOT"]).mkdir(parents=True, exist_ok=True)

    return persist


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """
    Minimal DataFrame med 'ts' på 2025-01-02 (UTC), så rotate_partition kan
    udlede korrekt dags-partition uden at testen skal angive dato.
    """
    base = pd.Timestamp("2025-01-02T00:00:00Z")
    return pd.DataFrame(
        {
            "ts": [base + pd.Timedelta(seconds=i) for i in (1, 2, 3)],
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10.0, 11.0, 12.0],
        }
    )
