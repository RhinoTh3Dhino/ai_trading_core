import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.project_path import PROJECT_ROOT  # AUTO PATH CONVERTED

def generate_trend_graph(
    history_path=PROJECT_ROOT / "outputs" / "performance_history.csv",  # AUTO PATH CONVERTED
    img_path=PROJECT_ROOT / "outputs" / "balance_trend.png",            # AUTO PATH CONVERTED
    title="Balanceudvikling over tid",
    xlabel="Tid",
    ylabel="Balance",
    legend_title="Symbol"
):
    """
    Genererer og gemmer en trend-graf over balance for hvert aktiv/symbol over tid.
    Hvis performance_history.csv ikke findes eller er tom, oprettes en dummy-graf.
    """
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    if not os.path.exists(history_path):
        print(f"[WARN] Historik-fil findes ikke: {history_path}. Opretter dummy-graf.")
        # Dummy-data
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
            "Navn": ["Ingen data"],
            "Balance": [0]
        })
    else:
        try:
            df = pd.read_csv(history_path)
        except Exception as e:
            print(f"[WARN] Kunne ikke indlæse {history_path}: {e}. Opretter dummy-graf.")
            df = pd.DataFrame({
                "timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
                "Navn": ["Ingen data"],
                "Balance": [0]
            })
        # Sikring mod tom fil eller manglende kolonner
        if df.empty or not all(col in df.columns for col in ["timestamp", "Balance", "Navn"]):
            print("[WARN] Mangler nødvendige kolonner i performance_history.csv – opretter dummy-graf.")
            df = pd.DataFrame({
                "timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
                "Navn": ["Ingen data"],
                "Balance": [0]
            })

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
    plt.savefig(img_path)
    plt.close()
    print(f"[INFO] Trend-graf genereret: {img_path}")
    return img_path

# Eksempel på brug:
# generate_trend_graph()
