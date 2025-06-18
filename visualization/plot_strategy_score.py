import matplotlib.pyplot as plt

def plot_strategy_scores(strategy_metrics, save_path=None):
    labels = list(strategy_metrics.keys())
    profits = [m['profit_pct'] for m in strategy_metrics.values()]
    win_rates = [m['win_rate'] for m in strategy_metrics.values()]

    x = range(len(labels))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(x, profits, color='tab:blue', alpha=0.6, label='Profit %')
    ax2.plot(x, [w*100 for w in win_rates], color='tab:red', marker='o', label='Win-rate (%)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Profit (%)')
    ax2.set_ylabel('Win-rate (%)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Strategi-score pr. strategi')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
