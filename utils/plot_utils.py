import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_trend_graph(
    history_path="outputs/performance_history.csv",
    img_path="outputs/balance_trend.png",
    title="Balanceudvikling over tid",
    xlabel="Tid",
    ylabel="Balance",
    legend_title="Symbol"
):
    """
    Genererer og gemmer en trend-graf over balance (eller andet) for hver aktiv/symbol over tid.
    Kan bruges til Telegram, rapport eller dashboard.
    """
    if not os.path.exists(history_path):
        print(f"[WARN] Historik-fil findes ikke: {history_path}")
        return None

    df = pd.read_csv(history_path)
    if "timestamp" not in df.columns or "Balance" not in df.columns or "Navn" not in df.columns:
        print("[WARN] Mangler nødvendige kolonner i performance_history.csv (kræver 'timestamp', 'Balance', 'Navn')")
        return None

    plt.figure(figsize=(10, 6))
    for name in df['Navn'].unique():
        sub = df[df['Navn'] == name]
        plt.plot(sub['timestamp'], sub['Balance'], label=name, marker='o')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=legend_title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    print(f"[INFO] Trend-graf genereret: {img_path}")
    return img_path

# Eksempel på brug:
# generate_trend_graph()
