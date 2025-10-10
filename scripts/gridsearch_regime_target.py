# scripts/gridsearch_regime_target.py
"""
Gridsearch og performance-måling på regime-adaptiv target.
Kører alle feature-kombinationer og finder højeste winrate.
"""

import argparse
import itertools
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from utils.project_path import PROJECT_ROOT
from utils.telegram_utils import send_telegram_message


def calculate_sharpe(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    std = np.std(excess_returns)
    if std == 0:
        return 0.0
    sharpe_ratio = np.mean(excess_returns) / (std + 1e-9)
    return sharpe_ratio


def train_and_eval(df, feature_cols, target_col, test_size=0.4):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
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
    preds = (preds_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    winrate = report["1"]["recall"] if "1" in report else 0.0
    n_1 = np.sum(preds == 1)
    n_0 = np.sum(preds == 0)
    n_test = len(preds)
    returns = np.where(preds == 1, 1, -1)
    sharpe = calculate_sharpe(returns)
    # Ekstra debug
    cm = confusion_matrix(y_test, preds)
    return accuracy, winrate, sharpe, n_1, n_0, n_test, cm, report


def main():
    parser = argparse.ArgumentParser(
        description="Gridsearch regime-adaptiv target med LightGBM."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h_with_regime_target.csv"),
        help="Sti til target-CSV med regime targets.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_regime_adapt",
        help="Target-kolonne (default: regime).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="close,rsi_14,ema_9,macd,macd_signal,vwap,atr_14,regime",
        help="Kommasepareret feature-liste.",
    )
    parser.add_argument(
        "--max_features", type=int, default=7, help="Max features i kombination."
    )
    parser.add_argument("--test_size", type=float, default=0.4, help="Test split.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "gridsearch_regime_results.csv"),
        help="Hvor CSV-resultater skal gemmes.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip().str.lower()
    all_features = [col.strip().lower() for col in args.features.split(",")]

    # Fjern rækker med NaN i target eller features
    df = df.dropna(subset=[args.target] + all_features)

    results = []
    # Gridsearch over alle feature-kombinationer (1 ... max_features)
    for n_feats in range(1, args.max_features + 1):
        for feats in itertools.combinations(all_features, n_feats):
            try:
                acc, winrate, sharpe, n_1, n_0, n_test, cm, report = train_and_eval(
                    df, list(feats), args.target, test_size=args.test_size
                )
                results.append(
                    {
                        "features": list(feats),
                        "accuracy": acc,
                        "winrate": winrate,
                        "sharpe": sharpe,
                        "n_pred_1": n_1,
                        "n_pred_0": n_0,
                        "n_test": n_test,
                    }
                )
                print(
                    f"Features: {feats} | Accuracy: {acc:.3f} | Winrate: {winrate:.3f} | Sharpe: {sharpe:.2f} | n_1: {n_1}, n_0: {n_0}, n_test: {n_test}"
                )
            except Exception as e:
                print(f"[FEJL] {feats}: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["winrate", "sharpe"], ascending=[False, False]
    )
    print("\n=== Gridsearch resultater (top 10) ===")
    print(results_df.head(10))

    results_df.to_csv(args.output, index=False)
    print(f"[OK] Gridsearch regime-resultater gemt i: {args.output}")

    # Telegram-summary (kun bedste)
    if not results_df.empty:
        top = results_df.iloc[0]
        msg = (
            f"📊 Gridsearch Regime-Target\n"
            f"Bedste features: {top['features']}\n"
            f"Accuracy: {top['accuracy']:.2%}\n"
            f"Winrate: {top['winrate']:.2%}\n"
            f"Sharpe: {top['sharpe']:.2f}\n"
            f"n_1: {int(top['n_pred_1'])}, n_0: {int(top['n_pred_0'])} (Test: {int(top['n_test'])})"
        )
        send_telegram_message(msg)


if __name__ == "__main__":
    main()
