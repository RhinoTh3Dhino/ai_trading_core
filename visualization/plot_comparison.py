# visualization/plot_comparison.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_comparison(
    models_metrics: dict,
    metric_keys: list = None,
    title: str = "Performance Comparison (ML vs DL vs Ensemble)",
    save_path: str = None,
    figsize: tuple = (8, 5),
    legend_loc: str = "best"
):
    """
    Visualiserer og sammenligner performance for flere modeller (ML, DL, ensemble).

    Args:
        models_metrics (dict): Dict med modelnavn som key og dict med metrics som value,
                               fx {"ML": {...}, "DL": {...}, "Ensemble": {...}}
        metric_keys (list): Hvilke metrics der skal sammenlignes (fx ["accuracy", "profit_pct"])
        title (str): Plot-titel.
        save_path (str): Hvor plot gemmes (PNG). Hvis None, gemmes i "graphs/" med timestamp.
        figsize (tuple): Figur-størrelse.
        legend_loc (str): Legendens placering.

    Returns:
        str: Path til gemt plot.
    """
    # Find alle metrikker
    if metric_keys is None:
        # Brug alle der findes i dicts
        keys = set()
        for v in models_metrics.values():
            keys.update(v.keys())
        metric_keys = sorted(keys)
    model_names = list(models_metrics.keys())

    # Byg matrix: rows=model, cols=metric
    data = np.array([[models_metrics[m].get(k, np.nan) for k in metric_keys] for m in model_names])

    fig, ax = plt.subplots(figsize=figsize)
    width = 0.22 if len(model_names) > 2 else 0.3
    x = np.arange(len(metric_keys))
    for i, (mname, vals) in enumerate(zip(model_names, data)):
        ax.bar(x + (i - len(model_names)/2)*width + width/2, vals, width, label=mname)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=15)
    ax.set_ylabel("Score / pct. / value")
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Gem fil
    if save_path is None:
        os.makedirs("graphs", exist_ok=True)
        import datetime
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"graphs/model_comparison_{dt_str}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Model comparison gemt til: {save_path}")
    return save_path

# === CLI-brug/test ===
if __name__ == "__main__":
    # Demo: Sammenlign dummy performance
    models_metrics = {
        "ML": {"accuracy": 0.59, "profit_pct": 4.2, "win_rate": 0.47},
        "DL": {"accuracy": 0.62, "profit_pct": 7.1, "win_rate": 0.51},
        "Ensemble": {"accuracy": 0.64, "profit_pct": 8.4, "win_rate": 0.55}
    }
    plot_comparison(models_metrics, metric_keys=["accuracy", "profit_pct", "win_rate"])
