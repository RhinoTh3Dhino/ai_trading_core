import os
import matplotlib.pyplot as plt
import numpy as np

from utils.project_path import PROJECT_ROOT


def safe_str(x):
    """Sørger for at alle feature-navne er rene str – aldrig array eller liste."""
    # np.generic = fx numpy.str_ etc.
    if isinstance(x, (np.ndarray, list)):
        if len(x) == 1:
            return str(x[0])
        return str(x.tolist())
    if isinstance(x, np.generic):
        return str(x.item())
    return str(x)


def plot_feature_importance(
    feature_names,
    importance_scores,
    # AUTO PATH CONVERTED
    out_path=PROJECT_ROOT / "outputs" / "feature_importance.png",
    method="Permutation",
    top_n=None,
    show=False,
):
    """
    Visualiser og gem feature importance plot.
    Args:
        feature_names (iterable): Navne på features.
        importance_scores (iterable): Scores, samme rækkefølge som feature_names.
        out_path (str): Filnavn (png).
        method (str): 'Permutation', 'SHAP' eller andet til titel.
        top_n (int): Vis kun top N features (valgfrit).
        show (bool): Hvis True, vis plot direkte.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- ALTID robust konvertering ---
    if hasattr(feature_names, "to_numpy"):
        feature_names = feature_names.to_numpy()
    elif hasattr(feature_names, "values"):
        feature_names = np.array(feature_names.values)
    else:
        feature_names = np.array(feature_names)

    importance_scores = np.array(importance_scores)

    sorted_idx = importance_scores.argsort()[::-1]
    feature_names = feature_names[sorted_idx]
    importance_scores = importance_scores[sorted_idx]
    if top_n is not None:
        feature_names = feature_names[:top_n]
        importance_scores = importance_scores[:top_n]

    # --- KONVERTER ALT til rene str ---
    feature_names = [safe_str(x) for x in feature_names]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_scores)
    plt.xlabel("Feature Importance")
    plt.title(f"{method} Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()
    print(f"✅ Feature importance-plot gemt: {out_path}")


def plot_shap_importance(
    shap_values,
    feature_names,
    # AUTO PATH CONVERTED
    out_path=PROJECT_ROOT / "outputs" / "feature_importance_shap.png",
    top_n=None,
    show=False,
):
    """
    Visualiser og gem SHAP importance (mean absolut værdi pr feature).
    Args:
        shap_values: SHAP values array eller SHAP object (fra explainer).
        feature_names: Navne på features (samme rækkefølge).
        out_path: Filnavn (png).
        top_n: Vis kun top N.
        show: Vis plot direkte.
    """
    # --- ALTID robust konvertering ---
    if hasattr(feature_names, "to_numpy"):
        feature_names = feature_names.to_numpy()
    elif hasattr(feature_names, "values"):
        feature_names = np.array(feature_names.values)
    else:
        feature_names = np.array(feature_names)

    if hasattr(shap_values, "values"):
        shap_vals = np.abs(np.array(shap_values.values)).mean(axis=0)
    else:
        shap_vals = np.abs(np.array(shap_values)).mean(axis=0)

    plot_feature_importance(
        feature_names=feature_names,
        importance_scores=shap_vals,
        out_path=out_path,
        method="SHAP",
        top_n=top_n,
        show=show,
    )


# --- Eksempel på brug ---
if __name__ == "__main__":
    # Demo: Bruges kun hvis scriptet køres direkte
    feature_names = ["RSI", "MACD", "EMA21", "Close", "Volume"]
    importance_scores = [0.25, 0.12, 0.05, 0.33, 0.22]
    # AUTO PATH CONVERTED
    plot_feature_importance(
        feature_names,
        importance_scores,
        out_path=PROJECT_ROOT / "outputs" / "feature_importance_demo.png",
        method="Permutation",
        top_n=5,
        show=True,
    )
