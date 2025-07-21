
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from datetime import datetime


# Importér strategi- og ensemblefunktioner
from ensemble.majority_vote_ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

# === OUTPUT DIR ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Regime-funktioner (brug fra din backtest eller her) ---
def compute_regime(df, ema_col="ema_200", price_col="close"):
    if "regime" in df.columns:
        return df
    df["regime"] = np.where(
        df[price_col] > df[ema_col], "bull",
        np.where(df[price_col] < df[ema_col], "bear", "neutral")
    )
    return df

def regime_performance(trades_df, regime_col="regime"):
    if regime_col not in trades_df.columns:
        return {}
    grouped = trades_df.groupby(regime_col)
    results = {}
    for name, group in grouped:
        n = len(group)
        win_rate = (group['profit'] > 0).mean() if n > 0 and 'profit' in group.columns else 0
        profit_pct = group['profit'].sum() if 'profit' in group.columns else 0
        drawdown_pct = group['drawdown'].min() if 'drawdown' in group.columns else None
        results[name] = {
            "num_trades": n,
            "win_rate": win_rate,
            "profit_pct": profit_pct,
            "drawdown_pct": drawdown_pct
        }
    return results

def run_backtest(df, signals):
    # Minimal backtest – indsæt din fulde backtest hvis ønsket!
    df = df.copy()
    df["signal"] = signals
    trades = []
    balance = 1000
    position = None
    entry_price = None
    for i, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]
        regime = row["regime"]
        if position is None and signal == 1:
            position = "long"
            entry_price = price
            trades.append({"timestamp": row["timestamp"], "type": "BUY", "price": price, "regime": regime, "profit": 0, "drawdown": None})
        elif position == "long" and (signal == -1 or i == df.index[-1]):
            profit = price - entry_price if entry_price else 0
            trades.append({"timestamp": row["timestamp"], "type": "SELL", "price": price, "regime": regime, "profit": profit, "drawdown": None})
            position = None
            entry_price = None
    trades_df = pd.DataFrame(trades)
    # Dummy-drawdown
    trades_df["drawdown"] = np.random.uniform(-5, 0, size=len(trades_df))
    return trades_df

def plot_regime_performance(perf_dict, strategy_name, run_id, output_dir=OUTPUT_DIR):
    regimes = list(perf_dict.keys())
    win_rates = [perf_dict[reg]["win_rate"] for reg in regimes]
    profits = [perf_dict[reg]["profit_pct"] for reg in regimes]
    n_trades = [perf_dict[reg]["num_trades"] for reg in regimes]
    x = np.arange(len(regimes))

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.bar(x-0.2, win_rates, width=0.4, label="Win-rate")
    ax2 = ax1.twinx()
    ax2.bar(x+0.2, profits, width=0.4, color="orange", label="Profit pct")
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes)
    ax1.set_ylabel("Win-rate")
    ax2.set_ylabel("Profit pct")
    plt.title(f"{strategy_name} – Regime performance ({run_id})")
    fig.tight_layout()
    plt.legend()
    png_path = os.path.join(output_dir, f"regime_perf_{strategy_name}_{run_id}.png")
    plt.savefig(png_path)
    plt.close()
    return png_path

def save_regime_report(strategy_stats, run_id, output_dir=OUTPUT_DIR):
    md_path = os.path.join(output_dir, f"regime_report_{run_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Regime performance rapport ({run_id})\n\n")
        for strat, (stats, plot_path) in strategy_stats.items():
            f.write(f"## {strat}\n")
            f.write(f"![{strat}]({os.path.basename(plot_path)})\n\n")
            for regime, perf in stats.items():
                f.write(f"- {regime}: win-rate={perf['win_rate']:.2%}, profit={perf['profit_pct']:.2f}, trades={perf['num_trades']}\n")
            f.write("\n")
    print(f"Regime rapport gemt: {md_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.feature_path)
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    df = compute_regime(df)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    strategy_stats = {}

    # ML/dummy signals
    np.random.seed(42)
    ml_signals = np.random.choice([1, 0, -1], size=len(df))
    rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
    macd_signals = macd_cross_signals(df)
    ensemble_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals)

    for name, signals in [
        ("ML", ml_signals),
        ("RSI", rsi_signals),
        ("MACD", macd_signals),
        ("Ensemble", ensemble_signals),
    ]:
        trades_df = run_backtest(df, signals)
        stats = regime_performance(trades_df)
        plot_path = plot_regime_performance(stats, name, run_id)
        strategy_stats[name] = (stats, plot_path)
        print(f"{name}: {stats}")

    save_regime_report(strategy_stats, run_id)
    print(f"Regime-analyse gennemført. Se rapport og grafer i: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
