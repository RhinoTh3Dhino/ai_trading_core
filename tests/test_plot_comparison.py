# tests/test_plot_comparison.py


import shutil
import pandas as pd
import numpy as np



from visualization.plot_comparison import plot_comparison

def make_dummy_results():
    # Dummy backtest-metrics
    results = {
        "ML": {
            "profit_pct": 3.2,
            "win_rate": 0.55,
            "drawdown_pct": -1.2,
            "num_trades": 13
        },
        "DL": {
            "profit_pct": 7.8,
            "win_rate": 0.62,
            "drawdown_pct": -1.9,
            "num_trades": 15
        },
        "ENSEMBLE": {
            "profit_pct": 8.4,
            "win_rate": 0.64,
            "drawdown_pct": -1.1,
            "num_trades": 18
        }
    }
    return results

def test_plot_comparison_creates_image(tmp_path):
    """Test at plot_comparison() gemmer et billede og returnerer stien."""
    results = make_dummy_results()
    out_path = tmp_path / "test_comparison.png"
    output = plot_comparison(results, save_path=str(out_path))
    assert os.path.exists(output), "Output-billede blev ikke gemt"
    # Check at filen er større end 0 bytes
    assert os.path.getsize(output) > 100, "Gemte billede er for småt eller tomt"

def test_plot_comparison_runs_without_errors():
    """Test at plot_comparison() kan kaldes uden at fejle med standard-data."""
    results = make_dummy_results()
    # Brug et midlertidigt output-navn
    out_path = "test_output_comparison.png"
    try:
        output = plot_comparison(results, save_path=out_path)
        assert os.path.exists(output), "Plot blev ikke genereret"
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
