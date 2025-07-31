# analyze_strategies.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import platform
import subprocess

# Tilføj rodmappen til sys.path for at kunne importere dine egne moduler


# === OUTPUT DIR ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# === IMPORTÉR DINE EGNE FUNKTIONER HER ===
# from engine import run_backtest_for_all_strategies
# from telegram_utils import send_telegram_message, send_telegram_photo


# --- DUMMYDATA: Udskift med din egen backtest-funktion! ---
def dummy_backtest_all():
    return {
        "ML": {"win_rate": 0.42, "profit": 123, "drawdown": -32, "trades": 53},
        "RSI": {"win_rate": 0.36, "profit": 98, "drawdown": -45, "trades": 51},
        "MACD": {"win_rate": 0.39, "profit": 101, "drawdown": -41, "trades": 50},
        "Ensemble": {"win_rate": 0.47, "profit": 152, "drawdown": -28, "trades": 55},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse og visualisering af strategi-performance"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output-mappe til alle filer (default=outputs/)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Manuelt run-id (ellers auto)"
    )
    return parser.parse_args()


def save_strategy_scores(metrics, csv_path):
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    scores = []
    for strat, vals in metrics.items():
        row = {"run_id": run_id, "strategy": strat}
        row.update(vals)
        scores.append(row)
    df = pd.DataFrame(scores)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)
    return run_id


def plot_strategy_scores(metrics, run_id, out_path):
    strategies = list(metrics.keys())
    win_rates = [metrics[s]["win_rate"] for s in strategies]
    profits = [metrics[s]["profit"] for s in strategies]
    drawdowns = [metrics[s]["drawdown"] for s in strategies]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(strategies, win_rates, alpha=0.6, label="Win-rate", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(strategies, profits, "ro-", label="Profit")
    ax2.plot(strategies, drawdowns, "go-", label="Drawdown")
    ax1.set_ylabel("Win-rate")
    ax2.set_ylabel("Profit / Drawdown")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title(f"Strategi-score pr. run {run_id}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_strategy_history(csv_path, out_path):
    df = pd.read_csv(csv_path)
    if len(df["run_id"].unique()) < 2:
        return None  # For lidt historik
    pivot = df.pivot(index="run_id", columns="strategy", values="win_rate")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu")
    plt.title("Strategi win-rate over tid")
    plt.ylabel("Run")
    plt.xlabel("Strategi")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def format_strategy_message(metrics, run_id):
    msg = f"📊 *Strategi-score for seneste run ({run_id}):*\n"
    for strat, vals in metrics.items():
        msg += f"• {strat}: win-rate = {vals['win_rate']:.1%}, profit = {vals['profit']}, drawdown = {vals['drawdown']}, trades = {vals['trades']}\n"
    return msg


def update_botstatus_md(metrics, run_id, plot_path, heatmap_path=None, md_path=None):
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Strategi-score for run {run_id}\n")
        for strat, vals in metrics.items():
            f.write(
                f"- {strat}: win-rate={vals['win_rate']:.1%}, profit={vals['profit']}, drawdown={vals['drawdown']}, trades={vals['trades']}\n"
            )
        f.write(f"![Strategi-score]({os.path.basename(plot_path)})\n")
        if heatmap_path:
            f.write(f"![Score-heatmap]({os.path.basename(heatmap_path)})\n")


def save_run_markdown(
    metrics, run_id, plot_path, heatmap_path=None, output_dir=DEFAULT_OUTPUT_DIR
):
    md_path = os.path.join(output_dir, f"report_{run_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Strategi-score rapport – run {run_id}\n\n")
        for strat, vals in metrics.items():
            f.write(
                f"- **{strat}**: win-rate = {vals['win_rate']:.1%}, profit = {vals['profit']}, drawdown = {vals['drawdown']}, trades = {vals['trades']}\n"
            )
        f.write(f"\n![Strategi-score]({os.path.basename(plot_path)})\n")
        if heatmap_path:
            f.write(f"\n![Score-heatmap]({os.path.basename(heatmap_path)})\n")
    print(f"Markdown-rapport gemt: {md_path}")


def check_and_alert(metrics, run_id):
    for strat, vals in metrics.items():
        if vals["win_rate"] < 0.25:
            msg = f"🚨 ALERT: {strat} har lav win-rate ({vals['win_rate']:.0%}) i run {run_id}!"
            print(msg)
            # send_telegram_message(msg)
        if vals["drawdown"] < -40:
            msg = f"⚠️ Advarsel: {strat} drawdown ({vals['drawdown']}) i run {run_id}!"
            print(msg)
            # send_telegram_message(msg)


def open_output_folder(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"Kunne ikke åbne output-mappen automatisk: {e}")


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "strategy_scores.csv")
    md_path = os.path.join(output_dir, "BotStatus.md")

    # Brug rigtig backtest hvis ønsket:
    # metrics = run_backtest_for_all_strategies()
    metrics = dummy_backtest_all()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M")
    # CSV-log
    _ = save_strategy_scores(metrics, csv_path=csv_path)
    # Grafer
    plot_path = plot_strategy_scores(
        metrics,
        run_id,
        out_path=os.path.join(output_dir, f"strategy_scores_{run_id}.png"),
    )
    heatmap_path = plot_strategy_history(
        csv_path=csv_path,
        out_path=os.path.join(output_dir, "strategy_score_heatmap.png"),
    )
    # Telegram (valgfrit)
    telegram_msg = format_strategy_message(metrics, run_id)
    # send_telegram_message(telegram_msg)
    # send_telegram_photo(plot_path)
    # if heatmap_path:
    #     send_telegram_photo(heatmap_path)
    # Advarsler
    check_and_alert(metrics, run_id)
    # Opdater BotStatus.md og lav separat markdown-rapport
    update_botstatus_md(metrics, run_id, plot_path, heatmap_path, md_path=md_path)
    save_run_markdown(metrics, run_id, plot_path, heatmap_path, output_dir)
    # Åbn output-folder automatisk
    open_output_folder(output_dir)
    print(f"Analyse gennemført. Se alle outputs i: {output_dir}")


if __name__ == "__main__":
    main()
