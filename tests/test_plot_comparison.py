# tests/test_plot_comparison.py

import sys
import os
from pathlib import Path

# Sikrer projekt-roden er på sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Brug Agg backend i testmiljø for at undgå GUI-afhængigheder (Tkinter)
import matplotlib
matplotlib.use("Agg")

import shutil
import pandas as pd
import numpy as np

from visualization.plot_comparison import plot_comparison


def make_dummy_results():
    """Returnerer et dummy-sæt med model-metrics til test."""
    return {
        "ML": {
            "profit_pct": 3.2,
            "win_rate": 0.55,
            "drawdown_pct": -1.2,
            "num_trades": 13,
        },
        "DL": {
            "profit_pct": 7.8,
            "win_rate": 0.62,
            "drawdown_pct": -1.9,
            "num_trades": 15,
        },
        "ENSEMBLE": {
            "profit_pct": 8.4,
            "win_rate": 0.64,
            "drawdown_pct": -1.1,
            "num_trades": 18,
        },
    }


def test_plot_comparison_creates_image(tmp_path):
    """Test at plot_comparison() gemmer et billede og returnerer stien."""
    results = make_dummy_results()
    out_path = tmp_path / "test_comparison.png"
    output = plot_comparison(results, save_path=str(out_path))
    assert os.path.exists(output), "Output-billede blev ikke gemt"
    assert os.path.getsize(output) > 100, "Gemte billede er for småt eller tomt"


def test_plot_comparison_runs_without_errors(tmp_path):
    """Test at plot_comparison() kan kaldes uden at fejle med standard-data."""
    results = make_dummy_results()
    out_path = tmp_path / "test_output_comparison.png"
    output = plot_comparison(results, save_path=str(out_path))
    assert os.path.exists(output), "Plot blev ikke genereret"
    assert os.path.getsize(output) > 100, "Plot filen blev genereret men er tom"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
