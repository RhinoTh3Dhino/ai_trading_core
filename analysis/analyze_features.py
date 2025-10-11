# Brug Agg-backend FØR import af matplotlib!
import matplotlib

matplotlib.use("Agg")
# Skjul joblib/loky resource_tracker-advarsler globalt
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# === OUTPUT DIR ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MODEL & DATA IMPORTS ===
# For demo: Simuleret model/data
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def load_dummy_data():
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "ema_9": np.random.normal(100, 2, 200),
            "ema_21": np.random.normal(102, 2, 200),
            "ema_50": np.random.normal(105, 2, 200),
            "rsi_14": np.random.uniform(10, 90, 200),
            "macd": np.random.normal(0, 1, 200),
            "macd_signal": np.random.normal(0, 1, 200),
            "atr_14": np.random.uniform(0.5, 2, 200),
            "volume": np.random.uniform(100, 200, 200),
        }
    )
    y = (X["rsi_14"] > 60).astype(int)  # dummy buy/sell-label
    return X, y


def load_trained_rf_model(X, y):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


# === FEATURE IMPORTANCE: CLASSIC (f.eks. RandomForest/XGBoost) ===
def classic_feature_importance(model, feature_names, run_id, output_dir=OUTPUT_DIR):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    feat_df = feat_df.sort_values("importance", ascending=False)
    csv_path = os.path.join(output_dir, f"feature_importance_{run_id}_classic.csv")
    feat_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df["feature"], feat_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Classic Feature Importance ({run_id})")
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"feature_importance_{run_id}_classic.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Classic feature importance gemt som: {csv_path} og {png_path}")
    return feat_df, csv_path, png_path


# === FEATURE IMPORTANCE: PERMUTATION (sklearn, ML og NN) ===
def permutation_feature_importance(model, X, y, run_id, output_dir=OUTPUT_DIR):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    feat_df = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
    feat_df = feat_df.sort_values("importance", ascending=False)
    csv_path = os.path.join(output_dir, f"feature_importance_{run_id}_permutation.csv")
    feat_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df["feature"], feat_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Permutation Feature Importance ({run_id})")
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"feature_importance_{run_id}_permutation.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Permutation importance gemt som: {csv_path} og {png_path}")
    return feat_df, csv_path, png_path


# === FEATURE IMPORTANCE: SHAP (ML og NN, nu med TreeExplainer til RF/XGB) ===
def shap_feature_importance(model, X, run_id, output_dir=OUTPUT_DIR):
    try:
        import shap

        # Bruger TreeExplainer til tree-models (RandomForest, XGBoost, LightGBM)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        png_path = os.path.join(output_dir, f"feature_importance_{run_id}_shap.png")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"SHAP feature importance gemt som: {png_path}")
        return png_path
    except ImportError:
        print("SHAP ikke installeret, springer SHAP-analyse over.")
        return None
    except Exception as e:
        print(f"SHAP-fejl: {e}")
        return None


# === GEM MARKDOWN RAPPORT ===
def save_feature_report(
    run_id, classic_png=None, permutation_png=None, shap_png=None, output_dir=OUTPUT_DIR
):
    md_path = os.path.join(output_dir, f"feature_report_{run_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Feature Importance Rapport ({run_id})\n\n")
        if classic_png:
            f.write(f"## Classic importance\n\n![classic]({os.path.basename(classic_png)})\n\n")
        if permutation_png:
            f.write(
                f"## Permutation importance\n\n![permutation]({os.path.basename(permutation_png)})\n\n"
            )
        if shap_png:
            f.write(f"## SHAP importance\n\n![shap]({os.path.basename(shap_png)})\n\n")
    print(f"Markdown-feature-rapport gemt: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-importance analyse og visualisering")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classic", "permutation", "shap", "all"],
        default="all",
        help="Vælg importance-metode (classic|permutation|shap|all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    X, y = load_dummy_data()  # Erstat med din egen data-loader!
    model = load_trained_rf_model(X, y)  # Erstat evt. med din egen model!

    classic_png = permutation_png = shap_png = None

    if args.mode in ["classic", "all"]:
        feat_df, csv_path, classic_png = classic_feature_importance(
            model, X.columns, run_id, output_dir=OUTPUT_DIR
        )
    if args.mode in ["permutation", "all"]:
        feat_df2, csv_path2, permutation_png = permutation_feature_importance(
            model, X, y, run_id, output_dir=OUTPUT_DIR
        )
    if args.mode in ["shap", "all"]:
        shap_png = shap_feature_importance(model, X, run_id, output_dir=OUTPUT_DIR)

    save_feature_report(run_id, classic_png, permutation_png, shap_png, output_dir=OUTPUT_DIR)
    print(f"Feature-importance analyse gennemført ({args.mode}). Se alle outputs i: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
