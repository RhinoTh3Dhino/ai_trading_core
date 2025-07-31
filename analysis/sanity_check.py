# analysis/sanity_check.py
"""
Sanity check: Sammenlign model-performance med naive baselines (altid long/short/random).
Visualiserer fordeling, confusion matrix og edge mod simple strategier.

Brug:
python analysis/sanity_check.py --test_targets outputs/y_test.csv --predictions outputs/y_preds.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_confusion(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["SHORT (0)", "LONG (1)"])
    ax.set_yticklabels(["SHORT (0)", "LONG (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.colorbar(im)
    # Skriv tal i felterne
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: sammenlign model mod naive baselines."
    )
    parser.add_argument(
        "--test_targets",
        type=str,
        required=True,
        help="Path til y_test.csv (targets fra test split)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path til y_preds.csv (model predictions på test split)",
    )
    args = parser.parse_args()

    # Indlæs data (forventet én kolonne, ingen header)
    y_test = pd.read_csv(args.test_targets, header=None).iloc[:, 0].astype(int).values
    preds = pd.read_csv(args.predictions, header=None).iloc[:, 0].astype(int).values

    always_long = np.ones_like(y_test)
    always_short = np.zeros_like(y_test)
    random_guess = np.random.choice([0, 1], size=len(y_test))

    # Edge (i pct): Modelens accuracy minus bedste baseline
    acc_model = accuracy_score(y_test, preds)
    acc_long = accuracy_score(y_test, always_long)
    acc_short = accuracy_score(y_test, always_short)
    acc_random = accuracy_score(y_test, random_guess)
    edge = acc_model - max(acc_long, acc_short, acc_random)

    print("\n=== Model performance ===")
    print("Accuracy:", acc_model)
    print(classification_report(y_test, preds, zero_division=0))

    print("\n=== Baseline: Always LONG ===")
    print("Accuracy:", acc_long)
    print(classification_report(y_test, always_long, zero_division=0))

    print("\n=== Baseline: Always SHORT ===")
    print("Accuracy:", acc_short)
    print(classification_report(y_test, always_short, zero_division=0))

    print("\n=== Baseline: RANDOM GUESSING ===")
    print("Accuracy:", acc_random)
    print(classification_report(y_test, random_guess, zero_division=0))

    print("\n=== EDGE (model minus bedste baseline) ===")
    print(f"Edge: {edge:.4f} ({edge*100:.2f} %-point over naive strategi)\n")

    # Visualisering af prediction-fordeling
    plt.figure(figsize=(8, 4))
    plt.hist([preds, y_test], bins=3, label=["Preds", "y_test"], rwidth=0.7)
    plt.title("Fordeling af model-predictions vs. faktiske targets")
    plt.legend()
    plt.show()

    # Confusion matrix plot
    plot_confusion(y_test, preds, title="Model - Confusion Matrix")

    # Plot fordelingen over tid
    plt.figure(figsize=(12, 3))
    plt.plot(y_test, label="y_test", alpha=0.7, linewidth=1)
    plt.plot(preds, label="preds", alpha=0.7, linewidth=1)
    plt.title("Predictions vs. targets (over tid)")
    plt.legend()
    plt.show()

    # Bonus: Udskriv fordeling og baseline-forventning
    print("=== Fordeling i test-targets ===")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(y_test)
        print(f"Klasse {u}: {c} ({pct:.1f}%)")
    print("\nHvis én klasse dominerer markant, vil naive strategier være svære at slå!")


if __name__ == "__main__":
    main()
