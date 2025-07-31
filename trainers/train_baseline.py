# trainers/train_baseline.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import itertools

from utils.project_path import PROJECT_ROOT
from utils.telegram_utils import send_telegram_message
import matplotlib.pyplot as plt
import os


def calculate_sharpe(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)
    return sharpe_ratio


def train_lightgbm_run(
    df, feature_cols, params, threshold=0.5, test_size=0.4, random_state=42
):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.dropna(subset=["target"], inplace=True)
    feature_cols_clean = [col.strip().lower() for col in feature_cols]

    X = df[feature_cols_clean]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)],
    )

    preds_prob = model.predict(X_test, num_iteration=model.best_iteration)
    preds = (preds_prob > threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    winrate = report["1"]["recall"] if "1" in report else 0.0

    returns = np.where((preds == 1), 1, -1)
    sharpe = calculate_sharpe(returns)

    return {
        "model": model,
        "accuracy": accuracy,
        "winrate": winrate,
        "sharpe": sharpe,
        "report": report,
        "params": params,
        "threshold": threshold,
    }


def gridsearch_baseline(filepath, feature_cols):
    # Param grid: Tilføj flere parametre eller thresholds efter behov
    param_grid = {
        "num_leaves": [15, 31],
        "learning_rate": [0.05, 0.1],
        "threshold": [0.4, 0.5, 0.6],
    }
    fixed_params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
    df = pd.read_csv(filepath)
    results = []
    best_score = 0
    best_run = None

    # Cartesian product of grid
    for num_leaves, learning_rate, threshold in itertools.product(
        param_grid["num_leaves"], param_grid["learning_rate"], param_grid["threshold"]
    ):
        params = fixed_params.copy()
        params["num_leaves"] = num_leaves
        params["learning_rate"] = learning_rate

        res = train_lightgbm_run(df, feature_cols, params, threshold)
        results.append(
            {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "threshold": threshold,
                "accuracy": res["accuracy"],
                "winrate": res["winrate"],
                "sharpe": res["sharpe"],
            }
        )
        print(
            f"Run: leaves={num_leaves}, lr={learning_rate}, thr={threshold:.2f} | Winrate={res['winrate']:.2%} | Sharpe={res['sharpe']:.2f}"
        )

        # Gem bedste model
        if res["winrate"] > best_score:
            best_score = res["winrate"]
            best_run = res

    # Gem gridsearch-resultater
    results_df = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "outputs" / "gridsearch_baseline_results.csv"
    os.makedirs(out_csv.parent, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"✅ Gridsearch-resultater gemt: {out_csv}")

    # Gem og rapportér bedste model/run
    if best_run:
        print(
            f"\n🏆 Bedste run: params={best_run['params']}, threshold={best_run['threshold']} | Winrate={best_run['winrate']:.2%}, Sharpe={best_run['sharpe']:.2f}"
        )
        # Gem model
        model_path = PROJECT_ROOT / "models" / "baseline_model_gridsearch.txt"
        best_run["model"].save_model(str(model_path))
        # Feature importance plot
        plt.figure(figsize=(8, 5))
        lgb.plot_importance(best_run["model"])
        plt.title("Feature Importance - Baseline Model (Gridsearch)")
        plt.tight_layout()
        plt.savefig(
            PROJECT_ROOT / "outputs" / "feature_importance_baseline_gridsearch.png"
        )
        plt.close()
        print(f"✅ Feature importance gemt i outputs/")
        # Telegram summary
        msg = (
            f"🏆 Baseline Gridsearch Best Model\n"
            f"Params: {best_run['params']}\n"
            f"Threshold: {best_run['threshold']}\n"
            f"Accuracy: {best_run['accuracy']:.2%}\n"
            f"Winrate: {best_run['winrate']:.2%}\n"
            f"Sharpe: {best_run['sharpe']:.2f}"
        )
        send_telegram_message(msg)
    else:
        print("❌ Ingen run opnåede ønsket winrate.")


if __name__ == "__main__":
    filepath = PROJECT_ROOT / "data" / "BTCUSDT_1h_features.csv"
    feature_cols = ["close", "rsi_14", "ema_9", "macd", "macd_signal", "vwap", "atr_14"]

    # Gridsearch – kør og find bedste baseline!
    gridsearch_baseline(filepath, feature_cols)
