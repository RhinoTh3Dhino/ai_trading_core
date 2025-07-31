import matplotlib.pyplot as plt
import os


def plot_drawdown(balance_df, symbol="BTC", save_dir="graphs/"):
    os.makedirs(save_dir, exist_ok=True)
    cumulative_max = balance_df["balance"].cummax()
    drawdown = balance_df["balance"] - cumulative_max
    plt.figure(figsize=(12, 6))
    plt.plot(balance_df["timestamp"], drawdown, label="Drawdown")
    plt.title(f"Drawdown for {symbol}")
    plt.xlabel("Tid")
    plt.ylabel("Drawdown")
    plt.legend()
    plot_path = os.path.join(
        save_dir,
        f"{symbol.lower()}_drawdown_{balance_df['timestamp'].iloc[-1][:10]}.png",
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Drawdown-graf gemt: {plot_path}")
    return plot_path
