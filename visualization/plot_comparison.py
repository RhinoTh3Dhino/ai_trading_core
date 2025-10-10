# visualization/plot_comparison.py

import os

# Brug Agg backend for at undgå GUI-afhængighed (headless til test/CI)
import matplotlib

matplotlib.use("Agg")

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PERCENT_HINTS = ("pct", "rate", "drawdown")  # heuristik til %-annotering


def _is_percent_metric(name: str) -> bool:
    if not name:
        return False
    n = str(name).lower()
    return any(h in n for h in _PERCENT_HINTS)


def _fmt_value(val: float, metric_name: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if _is_percent_metric(metric_name):
        return f"{val:.1f}%"
    # Sharpe/Sortino/profit_factor m.fl. → 2 decimals
    return f"{val:.2f}"


def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def plot_comparison(
    models_metrics: dict,
    metric_keys: list = None,
    title: str = "Performance Comparison (ML vs DL vs Ensemble)",
    save_path: str = None,
    figsize: tuple = (9.5, 5.5),
    legend_loc: str = "best",
):
    """
    Visualiserer og sammenligner performance for flere modeller (ML, DL, Ensemble).

    Args:
        models_metrics (dict): {"ML": {...}, "DL": {...}, "Ensemble": {...}}
        metric_keys (list): fx ["profit_pct", "max_drawdown", "sharpe", "sortino"].
                            Hvis None, bruges unionen af nøgler.
        title (str): Plot-titel.
        save_path (str): PNG-sti. Hvis None, gemmes i graphs/ med timestamp.
        figsize (tuple): Figur-størrelse.
        legend_loc (str): Legendens placering.

    Returns:
        str: Sti til gemt plot.
    """
    if not isinstance(models_metrics, dict) or not models_metrics:
        # Lav et tomt plot for at undgå fejl i CI
        os.makedirs("graphs", exist_ok=True)
        save_path = (
            save_path or f"graphs/model_comparison_{datetime.now():%Y%m%d_%H%M%S}.png"
        )
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.text(0.5, 0.5, "Ingen data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Plot] Model comparison gemt til: {save_path}")
        return save_path

    # Afled metric_keys hvis ikke opgivet
    if metric_keys is None:
        keys = set()
        for v in models_metrics.values():
            if isinstance(v, dict):
                keys.update(v.keys())
        metric_keys = sorted(keys)

    model_names = list(models_metrics.keys())
    if not metric_keys:
        metric_keys = ["profit_pct"]

    # Byg data-matrix (rows=model, cols=metric)
    data = []
    for m in model_names:
        row = []
        src = models_metrics.get(m, {}) or {}
        for k in metric_keys:
            row.append(_coerce_float(src.get(k, np.nan)))
        data.append(row)
    data = np.array(data, dtype=float) if len(data) else np.empty((0, len(metric_keys)))

    # Setup fig
    fig, ax = plt.subplots(figsize=figsize)
    n_models = len(model_names)
    n_metrics = len(metric_keys)
    width = min(0.8 / max(n_models, 1), 0.28)  # dynamisk søjlebredde

    x = np.arange(n_metrics, dtype=float)
    # For fornuftig y-akse ved mix af negative/positive værdier
    finite_vals = data[np.isfinite(data)]
    if finite_vals.size:
        ymin = float(np.nanmin(finite_vals))
        ymax = float(np.nanmax(finite_vals))
    else:
        ymin, ymax = 0.0, 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    # Lidt margin
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.15 * yrange)

    # Tegn søjler
    bars_per_model = []
    for i, (mname, vals) in enumerate(zip(model_names, data)):
        # NaN → vis som 0 men annotér som N/A
        vals_plot = np.where(np.isfinite(vals), vals, 0.0)
        bar_x = x + (i - (n_models - 1) / 2) * width
        b = ax.bar(bar_x, vals_plot, width=width, label=mname)
        bars_per_model.append((b, vals))

        # Annotér værdier over (eller under) søjlen
        for rect, raw_val, mk in zip(b, vals, metric_keys):
            label = _fmt_value(raw_val, mk)
            # placer annotation lidt over/under top afh. af fortegn
            h = rect.get_height()
            y = rect.get_y() + h
            dy = 0.02 * yrange if h >= 0 else -0.04 * yrange
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, y),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8.5,
                rotation=0,
            )

    # Akser/labels
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=15)
    ax.set_ylabel("Score / pct. / value")
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    # Gem fil
    if save_path is None:
        os.makedirs("graphs", exist_ok=True)
        save_path = f"graphs/model_comparison_{datetime.now():%Y%m%d_%H%M%S}.png"

    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Model comparison gemt til: {save_path}")
    return save_path


# === CLI-brug/test ===
if __name__ == "__main__":
    # Demo: Sammenlign dummy performance
    models_metrics = {
        "ML": {
            "profit_pct": -12.3,
            "max_drawdown": -35.7,
            "sharpe": 0.0,
            "sortino": 0.0,
        },
        "DL": {
            "profit_pct": 48.9,
            "max_drawdown": -22.1,
            "sharpe": 1.05,
            "sortino": 1.44,
        },
        "Ensemble": {
            "profit_pct": 7.6,
            "max_drawdown": -15.2,
            "sharpe": 0.42,
            "sortino": 0.61,
        },
    }
    plot_comparison(
        models_metrics,
        metric_keys=["profit_pct", "max_drawdown", "sharpe", "sortino"],
        title="Performance Comparison (Demo)",
    )
