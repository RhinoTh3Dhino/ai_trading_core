import glob, json
import pandas as pd
import numpy as np
import matplotlib
from utils.project_path import PROJECT_ROOT  # AUTO PATH CONVERTED
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime



# === LÆS COINS FRA config/coins.json (fallback til default) ===
COINS_JSON_PATH = os.path.join("config", "coins.json")
if os.path.exists(COINS_JSON_PATH):
    try:
        with open(COINS_JSON_PATH, "r", encoding="utf-8") as f:
            coin_obj = json.load(f)
            if isinstance(coin_obj, dict) and "coins" in coin_obj:
                COINS = [c.upper() for c in coin_obj["coins"]]
            elif isinstance(coin_obj, list):
                COINS = [c.upper() for c in coin_obj]
            else:
                COINS = ["BTC", "ETH", "DOGE"]
    except Exception as e:
        print(f"⚠️ Kunne ikke læse config/coins.json ({e}), bruger standard-coins.")
        COINS = ["BTC", "ETH", "DOGE"]
else:
    COINS = ["BTC", "ETH", "DOGE"]

# === DATA/FIL-STANDARD ===
FEATURE_DIR = PROJECT_ROOT / "outputs" / "feature_data/"  # AUTO PATH CONVERTED
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H%M")
OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Importér strategi og backtest utils fra dine egne moduler ===
from ensemble.majority_vote_ensemble import majority_vote_ensemble
from strategies.rsi_strategy import rsi_rule_based_signals
from strategies.macd_strategy import macd_cross_signals

def compute_regime(df, ema_col="ema_200", price_col="close"):
    if "regime" in df.columns:
        return df
    df["regime"] = np.where(
        df[price_col] > df[ema_col], "bull",
        np.where(df[price_col] < df[ema_col], "bear", "neutral")
    )
    return df

def run_backtest(df, signals):
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
    trades_df["drawdown"] = np.random.uniform(-5, 0, size=len(trades_df))
    return trades_df

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
            "win_rate": float(win_rate) if pd.notnull(win_rate) else 0.0,
            "profit_pct": float(profit_pct) if pd.notnull(profit_pct) else 0.0,
            "drawdown_pct": float(drawdown_pct) if pd.notnull(drawdown_pct) else 0.0
        }
    return results

def aggregate_coin_metrics(coin, regime_stats_dict):
    data = {"Coin": coin}
    for strat, reg_stats in regime_stats_dict.items():
        for reg in ["bull", "bear"]:
            key = f"{strat}_{reg}_win"
            key2 = f"{strat}_{reg}_profit"
            win = reg_stats.get(reg, {}).get("win_rate", 0.0)
            profit = reg_stats.get(reg, {}).get("profit_pct", 0.0)
            data[key] = float(win) if win is not None else 0.0
            data[key2] = float(profit) if profit is not None else 0.0
    return data

def plot_portfolio_heatmap(df, run_id, output_dir=OUTPUT_DIR):
    heatmap_df = df.set_index("Coin")[[c for c in df.columns if c.endswith("_win")]]
    heatmap_df = heatmap_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Portfolio win-rate heatmap ({run_id})")
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"portfolio_winrate_heatmap_{run_id}.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Portfolio heatmap gemt: {png_path}")
    return png_path

def save_portfolio_report(df, run_id, heatmap_path, output_dir=OUTPUT_DIR):
    md_path = os.path.join(output_dir, f"portfolio_report_{run_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Portfolio performance rapport ({run_id})\n\n")
        f.write(f"![Portfolio heatmap]({os.path.basename(heatmap_path)})\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Portfolio rapport gemt: {md_path}")

def main():
    coin_results = []
    for coin in COINS:
        # === DEBUG: Print hvilket mønster du søger efter, og hvilke filer du finder ===
        pattern = os.path.join(FEATURE_DIR, f"{coin.lower()}*_features*.csv")
        feature_files = glob.glob(pattern)
        print(f"[DEBUG] Matcher for {coin}: {pattern}")
        print(f"[DEBUG] Fandt disse filer: {feature_files}")

        if not feature_files:
            print(f"❌ Ingen feature-fil fundet for {coin}!")
            continue
        feature_path = feature_files[-1]
        print(f"[DEBUG] Læser feature-fil for {coin}: {feature_path}")
        df = pd.read_csv(feature_path)

        if df.empty:
            print(f"[DEBUG] DataFrame for {coin} er TOM! Tjek input-data.")
            continue

        print(f"[DEBUG] DataFrame for {coin}: Kolonner={list(df.columns)} | Første rækker:\n{df.head(3)}")

        # Ekstra check for nødvendige kolonner
        for col in ["close", "ema_200"]:
            if col not in df.columns:
                print(f"[ADVARSEL] Kolonnen '{col}' mangler i feature-fil for {coin} ({feature_path})")
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
        df = compute_regime(df)
        np.random.seed(42)
        ml_signals = np.random.choice([1, 0, -1], size=len(df))
        rsi_signals = rsi_rule_based_signals(df, low=45, high=55)
        macd_signals = macd_cross_signals(df)
        ensemble_signals = majority_vote_ensemble(ml_signals, rsi_signals, macd_signals)
        regime_stats_dict = {}
        for name, signals in [
            ("ML", ml_signals),
            ("RSI", rsi_signals),
            ("MACD", macd_signals),
            ("Ensemble", ensemble_signals),
        ]:
            trades_df = run_backtest(df, signals)
            stats = regime_performance(trades_df)
            regime_stats_dict[name] = stats
        coin_results.append(aggregate_coin_metrics(coin, regime_stats_dict))

    df = pd.DataFrame(coin_results)
    csv_path = os.path.join(OUTPUT_DIR, f"portfolio_metrics_{RUN_ID}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Samlet portefølje-metrics gemt: {csv_path}")
    heatmap_path = plot_portfolio_heatmap(df, RUN_ID)
    save_portfolio_report(df, RUN_ID, heatmap_path)
    print(f"\nMulti-coin batch-analyse færdig! Se rapport og heatmap i: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()