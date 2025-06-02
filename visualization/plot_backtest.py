import matplotlib.pyplot as plt
import os

def plot_backtest(balance_df, symbol="BTC", save_dir="graphs/"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(balance_df['timestamp'], balance_df['balance'], label='Balance')
    plt.title(f"Balanceudvikling for {symbol}")
    plt.xlabel("Tid")
    plt.ylabel("Balance")
    plt.legend()
    plot_path = os.path.join(save_dir, f"{symbol.lower()}_balance_{balance_df['timestamp'].iloc[-1][:10]}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Balancegraf gemt: {plot_path}")
    return plot_path
