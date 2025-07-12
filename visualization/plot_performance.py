# visualization/plot_performance.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def plot_performance(
    balance_df,
    trades_df=None,
    symbol="BTCUSDT",
    model_name=None,  # <-- Dette bruges til at sætte "ML", "DL", "Ensemble"
    save_path=None,
    show_trades=True,
    title_extra=None,
    figsize=(14, 7),
    dpi=100,
):
    """
    Plotter balance-udvikling, drawdown og (valgfrit) trades for en strategi/model.

    Args:
        balance_df (pd.DataFrame): Skal indeholde 'timestamp', 'balance', 'drawdown'. 'close' (pris) er bonus.
        trades_df (pd.DataFrame, optional): Skal indeholde 'timestamp', 'type' (BUY/TP/SL), 'price' (entry/exit).
        symbol (str): Symbol/navn på instrument (vises i titel).
        model_name (str): Navn på strategi/model, fx "ML", "DL", "Ensemble".
        save_path (str): Hvor grafen gemmes (.png). Hvis None, gemmes i "graphs" med tidsstempel.
        show_trades (bool): Om BUY/TP/SL markeres på balancen.
        title_extra (str): Ekstra tekst i titel (fx "Backtest 2025-07-12").
        figsize (tuple): Størrelse på figuren.
        dpi (int): Opløsning.
    Returns:
        str: Path til gemt fil.
    """
    if not isinstance(balance_df, pd.DataFrame):
        raise ValueError("balance_df skal være en DataFrame")
    if not {"timestamp", "balance", "drawdown"}.issubset(balance_df.columns):
        raise ValueError("balance_df skal indeholde 'timestamp', 'balance', 'drawdown'")

    plt.figure(figsize=figsize, dpi=dpi)

    # Konverter tid hvis nødvendigt
    x = pd.to_datetime(balance_df["timestamp"])

    # Plot balance
    plt.plot(x, balance_df["balance"], label="Balance", color="#0057B8", linewidth=2)
    # Plot drawdown
    plt.fill_between(x, balance_df["balance"] + balance_df["drawdown"], balance_df["balance"],
                     color="red", alpha=0.15, label="Drawdown")

    # Plot trades (BUY/TP/SL) hvis trades_df gives
    if show_trades and trades_df is not None and len(trades_df) > 0:
        trades = trades_df.copy()
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        buys = trades[trades["type"].str.upper() == "BUY"]
        tps = trades[trades["type"].str.upper() == "TP"]
        sls = trades[trades["type"].str.upper() == "SL"]
        plt.scatter(buys["timestamp"], buys["balance"], marker="^", color="green", label="BUY", zorder=5)
        plt.scatter(tps["timestamp"], tps["balance"], marker="o", color="lime", label="TP", zorder=5)
        plt.scatter(sls["timestamp"], sls["balance"], marker="v", color="red", label="SL", zorder=5)

    # Plot evt. pris (overlay)
    if "close" in balance_df.columns:
        ax2 = plt.twinx()
        ax2.plot(x, balance_df["close"], color="grey", linestyle="--", linewidth=1, alpha=0.4, label="Pris")
        ax2.set_ylabel("Pris (USDT)", color="grey")
        ax2.tick_params(axis='y', labelcolor='grey')

    # Titel og labels
    title = f"{symbol} | "
    title += (model_name.upper() if model_name else "AI Model")
    title += " | Performance"
    if title_extra:
        title += " | " + str(title_extra)
    plt.title(title)
    plt.xlabel("Tid")
    plt.ylabel("Balance (USDT)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Gem/returner path
    if save_path is None:
        os.makedirs("graphs", exist_ok=True)
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"graphs/performance_{symbol}_{model_name}_{dt_str}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Gemte performance-plot til: {save_path}")
    return save_path

# === CLI-brug/test ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot performance for AI trading bot")
    parser.add_argument("--balance", type=str, required=True, help="Path til balance_df (CSV)")
    parser.add_argument("--trades", type=str, default=None, help="Path til trades_df (CSV)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--model_name", type=str, default="AI Model")
    args = parser.parse_args()

    balance_df = pd.read_csv(args.balance)
    trades_df = pd.read_csv(args.trades) if args.trades else None
    plot_performance(
        balance_df,
        trades_df=trades_df,
        symbol=args.symbol,
        model_name=args.model_name,
        title_extra="CLI Test"
    )
