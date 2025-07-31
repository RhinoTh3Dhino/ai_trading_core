import matplotlib.pyplot as plt
import pandas as pd
import os

from utils.project_path import PROJECT_ROOT


# AUTO PATH CONVERTED
def plot_backtest(
    balance_csv=PROJECT_ROOT / "data" / "balance.csv", symbol="BTC", save_dir="graphs/"
):
    # Læs balance-data
    if isinstance(balance_csv, str):
        if not os.path.exists(balance_csv):
            print(f"❌ Filen {balance_csv} findes ikke!")
            return None
        balance_df = pd.read_csv(balance_csv)
    else:
        balance_df = balance_csv.copy()

    if balance_df.empty or "timestamp" not in balance_df or "balance" not in balance_df:
        print(
            "❌ balance_df mangler nødvendige kolonner ('timestamp', 'balance') eller er tom."
        )
        return None

    # Konverter timestamp til datetime, hvis muligt
    try:
        balance_df["timestamp"] = pd.to_datetime(balance_df["timestamp"])
    except Exception:
        pass

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(
        balance_df["timestamp"],
        balance_df["balance"],
        label="Balance",
        color="royalblue",
    )
    plt.title(f"Balanceudvikling for {symbol}")
    plt.xlabel("Tid")
    plt.ylabel("Balance")
    plt.grid(True, alpha=0.2)
    plt.legend()
    # Brug dato for sidste datapunkt til filnavn
    date_label = str(balance_df["timestamp"].iloc[-1])[:10].replace("-", "")
    plot_path = os.path.join(save_dir, f"{symbol.lower()}_balance_{date_label}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Balancegraf gemt: {plot_path}")
    return plot_path


if __name__ == "__main__":
    # Kør direkte fra terminal:
    # AUTO PATH CONVERTED
    plot_backtest(
        balance_csv=PROJECT_ROOT / "data" / "balance.csv",
        symbol="BTC",
        save_dir="graphs/",
    )
