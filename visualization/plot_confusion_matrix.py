# visualization/plot_confusion_matrix.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize=None,
    title="Confusion Matrix",
    save_path=None,
    figsize=(6, 6),
    cmap="Blues",
    show_values=True,
    extra_text=None
):
    """
    Plotter confusion matrix for ML/DL/ensemble signaler.

    Args:
        y_true (array-like): Sande labels.
        y_pred (array-like): Modelens forudsagte labels.
        labels (list, optional): Klassenavne, fx ["SELL", "BUY"] eller ["0", "1"].
        normalize ({"true", "pred", "all"}, optional): Normaliseringstype.
        title (str): Titel på plot.
        save_path (str, optional): Hvor plot gemmes (PNG). Hvis None, gemmes i "graphs/" med timestamp.
        figsize (tuple): Figur-størrelse.
        cmap (str): Farveskema.
        show_values (bool): Om værdier vises på plottet.
        extra_text (str): Ekstra info til titel (fx modelnavn, acc, dato).

    Returns:
        str: Path til gemt plot.
    """
    if labels is None:
        unique = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        labels = [str(l) for l in unique]
    cm = confusion_matrix(y_true, y_pred, labels=[int(l) if l.isdigit() else l for l in labels], normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap=cmap, colorbar=True)
    if not show_values:
        # Fjern tekst hvis ikke ønsket (skjuler værdier)
        for _, txt in np.ndenumerate(ax.texts):
            txt.set_visible(False)
    t = title
    if extra_text:
        t += " | " + extra_text
    plt.title(t)
    plt.tight_layout()

    # Gem fil
    if save_path is None:
        os.makedirs("graphs", exist_ok=True)
        import datetime
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"graphs/confusion_matrix_{dt_str}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Confusion matrix gemt til: {save_path}")
    return save_path

# === CLI-brug/test ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot confusion matrix for AI trading bot")
    parser.add_argument("--y_true", type=str, required=True, help="Path til y_true (CSV/NPY/TXT eller kolonne i DF)")
    parser.add_argument("--y_pred", type=str, required=True, help="Path til y_pred (CSV/NPY/TXT eller kolonne i DF)")
    parser.add_argument("--labels", type=str, default=None, help="Kommasepareret labels, fx 'SELL,BUY'")
    parser.add_argument("--normalize", type=str, default=None, help="'true', 'pred', 'all', eller None")
    args = parser.parse_args()

    # Smart indlæsning (tillader både CSV/NPY og ren kolonne)
    def smart_load(path):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            return df.iloc[:,0].values
        elif path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".txt"):
            return np.loadtxt(path)
        else:
            raise ValueError("Ukendt filtype: "+path)
    y_true = smart_load(args.y_true)
    y_pred = smart_load(args.y_pred)
    labels = args.labels.split(",") if args.labels else None
    plot_confusion_matrix(y_true, y_pred, labels=labels, normalize=args.normalize)

