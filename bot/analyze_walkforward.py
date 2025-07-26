from utils.project_path import PROJECT_ROOT
# analyze_walkforward.py

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === Find og indlæs nyeste walkforward-summary ===
summary_files = sorted(
# AUTO PATH CONVERTED
    glob.glob(PROJECT_ROOT / "outputs" / "walkforward/walkforward_summary_*.csv"),
    key=os.path.getmtime,
)
if not summary_files:
    raise FileNotFoundError("Ingen walkforward-summary CSV fundet!")
csv_path = summary_files[-1]
print(f"🔎 Indlæser: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Rows: {len(df)}, Kolonner: {df.columns.tolist()}")

# === Analyse af nøgle-metrics ===
metrics = [
    "test_sharpe", "test_win_rate", "test_pct_profit", "test_max_drawdown",
    "test_calmar", "test_volatility", "test_buyhold_pct",
    "test_rolling_sharpe", "test_trade_duration", "test_regime_drawdown"
]
print("\n--- Samlet statistik for alle splits ---")
for m in metrics:
    if m in df.columns:
        print(f"{m:<20} | Mean: {df[m].mean():>8.2f} | Min: {df[m].min():>8.2f} | Max: {df[m].max():>8.2f}")

# === Top-5 vinduer efter test Sharpe og pct. profit ===
cols_show = ["symbol", "timeframe", "window_start", "window_end", "test_sharpe",
             "test_win_rate", "test_pct_profit", "test_buyhold_pct"]
print("\n--- Top-5 vinduer efter test_sharpe ---")
print(df.sort_values("test_sharpe", ascending=False).head(5)[cols_show])
print("\n--- Top-5 vinduer efter test_pct_profit ---")
print(df.sort_values("test_pct_profit", ascending=False).head(5)[cols_show])

# === Plot Sharpe for alle splits (alle coins) ===
plt.figure(figsize=(10,5))
for symbol in df["symbol"].unique():
    sub = df[df["symbol"] == symbol]
    plt.plot(sub["window_start"], sub["test_sharpe"], marker='o', label=f"{symbol} Sharpe")
plt.xlabel("Walkforward Window Start")
plt.ylabel("Test Sharpe Ratio")
plt.title("Test Sharpe Ratio på tværs af splits")
plt.legend()
plt.tight_layout()
# AUTO PATH CONVERTED
sharpe_plot = fPROJECT_ROOT / "outputs" / "walkforward/plot_sharpe_per_split_{datetime.now():%Y%m%d_%H%M%S}.png"
plt.savefig(sharpe_plot)
print(f"✅ Gemte plot: {sharpe_plot}")

# === Plot Winrate og Buy & Hold ===
plt.figure(figsize=(10,5))
for symbol in df["symbol"].unique():
    sub = df[df["symbol"] == symbol]
    plt.plot(sub["window_start"], sub["test_win_rate"], label=f"{symbol} Winrate (%)")
    if "test_buyhold_pct" in sub.columns:
        plt.plot(sub["window_start"], sub["test_buyhold_pct"], "--", label=f"{symbol} Buy&Hold (%)")
plt.xlabel("Walkforward Window Start")
plt.ylabel("Pct / Winrate")
plt.title("Winrate og Buy & Hold på tværs af splits")
plt.legend()
plt.tight_layout()
# AUTO PATH CONVERTED
winrate_plot = fPROJECT_ROOT / "outputs" / "walkforward/plot_winrate_buyhold_per_split_{datetime.now():%Y%m%d_%H%M%S}.png"
plt.savefig(winrate_plot)
print(f"✅ Gemte plot: {winrate_plot}")

# === Ekstra: Plot rolling sharpe, trade duration, regime drawdown hvis muligt ===
if "test_rolling_sharpe" in df.columns:
    plt.figure(figsize=(10,5))
    for symbol in df["symbol"].unique():
        sub = df[df["symbol"] == symbol]
        plt.plot(sub["window_start"], sub["test_rolling_sharpe"], label=f"{symbol} Rolling Sharpe")
    plt.xlabel("Walkforward Window Start")
    plt.ylabel("Rolling Sharpe")
    plt.title("Rolling Sharpe på tværs af splits")
    plt.legend()
    plt.tight_layout()
# AUTO PATH CONVERTED
    fname = fPROJECT_ROOT / "outputs" / "walkforward/plot_rolling_sharpe_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.savefig(fname)
    print(f"✅ Gemte plot: {fname}")

if "test_trade_duration" in df.columns:
    plt.figure(figsize=(10,5))
    for symbol in df["symbol"].unique():
        sub = df[df["symbol"] == symbol]
        plt.plot(sub["window_start"], sub["test_trade_duration"], label=f"{symbol} Trade Duration")
    plt.xlabel("Walkforward Window Start")
    plt.ylabel("Trade Duration (timer)")
    plt.title("Trade Duration på tværs af splits")
    plt.legend()
    plt.tight_layout()
# AUTO PATH CONVERTED
    fname = fPROJECT_ROOT / "outputs" / "walkforward/plot_trade_duration_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.savefig(fname)
    print(f"✅ Gemte plot: {fname}")

if "test_regime_drawdown" in df.columns:
    plt.figure(figsize=(10,5))
    for symbol in df["symbol"].unique():
        sub = df[df["symbol"] == symbol]
        plt.plot(sub["window_start"], sub["test_regime_drawdown"], label=f"{symbol} Regime DD")
    plt.xlabel("Walkforward Window Start")
    plt.ylabel("Regime Drawdown")
    plt.title("Regime Drawdown på tværs af splits")
    plt.legend()
    plt.tight_layout()
# AUTO PATH CONVERTED
    fname = fPROJECT_ROOT / "outputs" / "walkforward/plot_regime_drawdown_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.savefig(fname)
    print(f"✅ Gemte plot: {fname}")

# === Gem top5/top10 splits til CSV/Excel/JSON ===
for N, tag in [(5, "top5"), (10, "top10")]:
    topN = df.sort_values("test_sharpe", ascending=False).head(N)
# AUTO PATH CONVERTED
    topN_csv = fPROJECT_ROOT / "outputs" / "walkforward/{tag}_sharpe_splits_{datetime.now():%Y%m%d_%H%M%S}.csv"
    topN_excel = topN_csv.replace(".csv", ".xlsx")
    topN_json = topN_csv.replace(".csv", ".json")
    topN.to_csv(topN_csv, index=False)
    topN.to_excel(topN_excel, index=False)
    topN.to_json(topN_json, orient="records")
    print(f"✅ Gemte {tag} splits til: {topN_csv}, {topN_excel}, {topN_json}")

# === Eksempel på pivottabel: Mean test_sharpe per symbol/timeframe ===
if "symbol" in df.columns and "timeframe" in df.columns and "test_sharpe" in df.columns:
    pivot = pd.pivot_table(df, values="test_sharpe", index="symbol", columns="timeframe", aggfunc="mean")
    print("\nPivot: Mean test_sharpe per symbol/timeframe:\n", pivot)

print("\nFærdig med analyse! Du kan nu bruge df til videre visualisering, grid search eller AI.")