from utils.project_path import PROJECT_ROOT
# visualization/plot_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# AUTO PATH CONVERTED
def plot_metrics_over_time(csv_path=PROJECT_ROOT / "data" / "model_eval.csv", out_path=PROJECT_ROOT / "data" / "metrics_over_time.png"):
    # Tjek om filen findes
    if not os.path.exists(csv_path):
        print(f"❌ Filen {csv_path} findes ikke!")
        return

    # Læs data
    df = pd.read_csv(csv_path)
    # Tjek om nødvendige kolonner er til stede
    if "timestamp" not in df or "accuracy" not in df or "f1" not in df:
        print("❌ Mangler påkrævede kolonner i model_eval.csv")
        return

    # Plot metrics
    plt.figure(figsize=(10,5))
    plt.plot(df["timestamp"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["timestamp"], df["f1"], marker='x', label="F1-score")
    plt.title("Accuracy og F1-score over tid")
    plt.xlabel("Træningstidspunkt")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Graf over metrics gemt: {out_path}")

if __name__ == "__main__":
    plot_metrics_over_time()