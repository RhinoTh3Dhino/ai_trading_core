# scripts/gridsearch_train.py
"""
Kør gridsearch over alle TP/SL-targets eller valgfrit target i din target-fil.
Træner og evaluerer LightGBM (Scikit-learn API) for hver target-kolonne.
Logger, debugger og gemmer resultater til CSV.

Nu med:
- On-the-fly balancering (undersample/oversample direkte i gridsearch)
- Automatisk class weights (LightGBM)
- Diagnose-debug: Target-, y_train-, y_test- og prediction-fordeling (Chain-of-Thought)
- Eksport af y_test og preds til sanity check

Brug fx:
python run.py scripts/gridsearch_train.py --input data/BTCUSDT_1h_with_target.csv --features close,rsi_14 --target target_regime_adapt --balance undersample --class_weights auto --test_size 0.4
"""

import argparse
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.project_path import PROJECT_ROOT
from utils.telegram_utils import send_telegram_message


def balance_df(df, target, method="undersample", random_state=42, verbose=True):
    counts = df[target].value_counts()
    classes = counts.index.tolist()
    min_class = counts.min()
    max_class = counts.max()
    if verbose:
        print(f"Før balancering: {dict(counts)}")
    dfs = []
    if method == "undersample":
        n = min_class
        for c in classes:
            dfs.append(df[df[target] == c].sample(n=n, random_state=random_state))
        balanced = (
            pd.concat(dfs)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        if verbose:
            print(f"Undersamplet alle klasser til: {n}")
    elif method == "oversample":
        n = max_class
        for c in classes:
            dfs.append(
                df[df[target] == c].sample(n=n, replace=True, random_state=random_state)
            )
        balanced = (
            pd.concat(dfs)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        if verbose:
            print(f"Oversamplet alle klasser til: {n}")
    else:
        raise ValueError(f"Ukendt balanceringsmetode: {method}")
    after_counts = balanced[target].value_counts()
    print(f"Efter balancering: {dict(after_counts)}")
    return balanced


def get_class_weights(y):
    values = pd.Series(y).value_counts(normalize=True)
    w = {cls: 1 / v for cls, v in values.items()}
    norm = sum(w.values()) / len(w)
    w = {cls: weight / norm for cls, weight in w.items()}
    return w


def calculate_sharpe(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    std = np.std(excess_returns)
    if std == 0:
        return 0.0
    sharpe_ratio = np.mean(excess_returns) / (std + 1e-9)
    return sharpe_ratio


def get_target_columns(df, user_target=None):
    if user_target:
        if user_target not in df.columns:
            raise ValueError(f"Target '{user_target}' ikke fundet i dataframe!")
        return [user_target]
    return [
        col
        for col in df.columns
        if str(col).startswith("target_tp")
        or str(col).startswith("target_regime")
        or col == "target"
    ]


def train_and_eval(df, feature_cols, target_col, test_size=0.4, class_weights=None):
    X = df[feature_cols]
    y = df[target_col]
    # Diagnose: Target fordeling (hele datasættet)
    print("\nDEBUG – target fordeling i data:")
    print(pd.Series(y).value_counts(dropna=False))

    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    print("\nDEBUG – y_train fordeling:")
    print(pd.Series(y_train).value_counts(dropna=False))
    print("DEBUG – y_test fordeling:")
    print(pd.Series(y_test).value_counts(dropna=False))

    model = LGBMClassifier(
        n_estimators=500,
        class_weight=class_weights,
        objective="binary",
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(20), log_evaluation(50)],
    )
    preds = model.predict(X_test)
    print("\nDEBUG – preds fordeling:", pd.Series(preds).value_counts(dropna=False))
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    winrate = report.get("1", {}).get("recall", 0.0)
    returns = np.where(preds == 1, 1, -1)
    sharpe = calculate_sharpe(returns)

    n_1 = int((preds == 1).sum())
    n_0 = int((preds == 0).sum())
    n_test = len(preds)
    cm = confusion_matrix(y_test, preds)
    print(f"[DEBUG] Prediction fordeling - 1: {n_1}, 0: {n_0}, test size: {n_test}")
    print("[DEBUG] Confusion Matrix:\n", cm)
    print(
        "[DEBUG] Classification report:\n",
        classification_report(y_test, preds, zero_division=0),
    )

    # --- EKSPORT AF Y_TEST OG PREDS ---
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    pd.Series(y_test).to_csv(
        os.path.join(outputs_dir, "y_test.csv"), index=False, header=False
    )
    pd.Series(preds).to_csv(
        os.path.join(outputs_dir, "y_preds.csv"), index=False, header=False
    )
    print(f"[OK] Gemte sanity check-filer: {outputs_dir}/y_test.csv & y_preds.csv")

    return accuracy, winrate, sharpe, n_1, n_0, n_test


def main():
    parser = argparse.ArgumentParser(
        description="Gridsearch TP/SL eller custom target med LightGBM (med on-the-fly balancing og class weights)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h_with_target.csv"),
        help="Sti til target-CSV med alle TP/SL targets.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="close,rsi_14,ema_9,macd,macd_signal,vwap,atr_14",
        help="Kommasepareret feature-liste.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Navn på target-kolonne (fx 'target_regime_adapt')",
    )
    parser.add_argument("--test_size", type=float, default=0.4, help="Test split.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "gridsearch_results.csv"),
        help="Hvor CSV-resultater skal gemmes.",
    )
    parser.add_argument(
        "--balance",
        type=str,
        default=None,
        choices=[None, "undersample", "oversample"],
        help="Balancér targets on-the-fly.",
    )
    parser.add_argument(
        "--class_weights",
        type=str,
        default=None,
        choices=[None, "auto"],
        help="Tilføj class weights i LightGBM (auto).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()
    feature_cols = [col.strip() for col in args.features.split(",")]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"[FEJL] Mangler features i data: {missing}")
        return

    target_cols = get_target_columns(df, args.target)
    print(f"[INFO] Fundet {len(target_cols)} target-kolonner: {target_cols}")

    results = []
    for target_col in target_cols:
        print(f"\n[INFO] Træner på target: {target_col}")
        df_tmp = df.dropna(subset=[target_col] + feature_cols)
        if len(df_tmp) < 500:
            print(
                f"[ADVARSEL] For få rækker ({len(df_tmp)}) for target {target_col}, springer over."
            )
            continue

        if args.balance:
            print(f"[INFO] Balancerer target med metode: {args.balance}")
            df_tmp = balance_df(df_tmp, target_col, method=args.balance)

        class_weights = None
        if args.class_weights == "auto":
            class_weights = get_class_weights(df_tmp[target_col])
            print(f"[INFO] Bruger class weights: {class_weights}")

        try:
            acc, winrate, sharpe, n_1, n_0, n_test = train_and_eval(
                df_tmp,
                feature_cols,
                target_col,
                test_size=args.test_size,
                class_weights=class_weights,
            )
            results.append(
                {
                    "target": target_col,
                    "accuracy": acc,
                    "winrate": winrate,
                    "sharpe": sharpe,
                    "n_pred_1": n_1,
                    "n_pred_0": n_0,
                    "n_test": n_test,
                    "n": len(df_tmp),
                }
            )
            print(
                f"  -> Accuracy: {acc:.3f} | Winrate: {winrate:.3f} | Sharpe: {sharpe:.2f} | n_1: {n_1}, n_0: {n_0}"
            )
        except Exception as e:
            print(f"[FEJL] Under træning på {target_col}: {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["winrate", "sharpe"], ascending=[False, False]
        )
        print("\n=== Gridsearch resultater (top 5) ===")
        print(results_df.head(5))
        results_df.to_csv(args.output, index=False)
        print(f"[OK] Gridsearch resultater gemt i: {args.output}")
        top = results_df.iloc[0]
        msg = (
            f"📊 Gridsearch (targets) – Bedste:\n"
            f"{top['target']}\n"
            f"Accuracy: {top['accuracy']:.2%}\n"
            f"Winrate: {top['winrate']:.2%}\n"
            f"Sharpe: {top['sharpe']:.2f}\n"
            f"n_1: {int(top['n_pred_1'])}, n_0: {int(top['n_pred_0'])} (Test: {int(top['n_test'])})"
        )
        send_telegram_message(msg)
    else:
        print("[ADVARSEL] Ingen modeller kunne trænes! Tjek targets og features.")


if __name__ == "__main__":
    main()
