
# Brug Agg-backend FØR import af matplotlib!
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Skjul joblib/loky resource_tracker-advarsler globalt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")



import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === OUTPUT DIR ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dummy_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'ema_9': np.random.normal(100, 2, 200),
        'ema_21': np.random.normal(102, 2, 200),
        'ema_50': np.random.normal(105, 2, 200),
        'rsi_14': np.random.uniform(10, 90, 200),
        'macd': np.random.normal(0, 1, 200),
        'macd_signal': np.random.normal(0, 1, 200),
        'atr_14': np.random.uniform(0.5, 2, 200),
        'volume': np.random.uniform(100, 200, 200)
    })
    y = (X['rsi_14'] > 60).astype(int)  # dummy buy/sell-label
    return X, y

def get_feature_ranking(X, y):
    # Træn fuld model, brug klassisk importance til sortering
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    feat_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feat_df = feat_df.sort_values('importance', ascending=False)
    return feat_df['feature'].tolist(), feat_df

def prune_and_evaluate(X, y, ranking, top_n_list, run_id, output_dir=OUTPUT_DIR):
    scores = []
    for n in top_n_list:
        top_feats = ranking[:n]
        X_pruned = X[top_feats]
        # Train/test split på hver iteration (kan evt. gøres én gang globalt hvis ønsket)
        X_train, X_test, y_train, y_test = train_test_split(X_pruned, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append({'n_features': n, 'features': top_feats, 'accuracy': acc})
        print(f"[{run_id}] Top {n} features: acc={acc:.4f} ({', '.join(top_feats)})")
    df = pd.DataFrame(scores)
    csv_path = os.path.join(output_dir, f'feature_pruning_{run_id}.csv')
    df.to_csv(csv_path, index=False)
    return df, csv_path

def plot_pruning_results(df, run_id, output_dir=OUTPUT_DIR):
    plt.figure(figsize=(8, 5))
    plt.plot(df['n_features'], df['accuracy'], 'o-', label='Test Accuracy')
    plt.xlabel("Antal features (top N)")
    plt.ylabel("Test accuracy")
    plt.title(f'Feature-pruning performance ({run_id})')
    plt.grid(True)
    plt.xticks(df['n_features'])
    plt.legend()
    png_path = os.path.join(output_dir, f'feature_pruning_{run_id}.png')
    plt.savefig(png_path)
    plt.close()
    print(f"Feature-pruning plot gemt: {png_path}")
    return png_path

def save_pruning_report(run_id, df, plot_path, output_dir=OUTPUT_DIR):
    md_path = os.path.join(output_dir, f'feature_pruning_report_{run_id}.md')
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Feature-pruning analyse ({run_id})\n\n")
        f.write("## Accuracy vs. antal features\n\n")
        f.write(f"![feature pruning]({os.path.basename(plot_path)})\n\n")
        f.write("## Data\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Feature-pruning rapport gemt: {md_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Auto feature-pruning/backtest pipeline")
    parser.add_argument("--steps", nargs='+', type=int, default=[3, 5, 8, 10],
                        help="Antal features at teste (fx 3 5 8 10)")
    return parser.parse_args()

def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    X, y = load_dummy_data()  # Skift til din egen data-loader i produktion!
    ranking, feat_df = get_feature_ranking(X, y)

    print(f"[{run_id}] Feature ranking: {ranking}")

    df, csv_path = prune_and_evaluate(X, y, ranking, args.steps, run_id, output_dir=OUTPUT_DIR)
    plot_path = plot_pruning_results(df, run_id, output_dir=OUTPUT_DIR)
    save_pruning_report(run_id, df, plot_path, output_dir=OUTPUT_DIR)

    print(f"Feature-pruning analyse færdig! Alt gemt i {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
