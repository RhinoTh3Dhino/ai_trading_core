"""
models/train_xgboost.py

Træner en XGBoost-model til trading-signaler (klassifikation)
- Understøtter klassisk træning og MLflow experiment tracking
- Gemmer og loader model (.json), log og metrics
- Automatisk feature selection, early stopping og artefakt-logging
"""

import os

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# === MLflow ===
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# --- Importer MLflow-utilities hvis muligt ---
try:
    from utils.mlflow_utils import setup_mlflow, start_mlflow_run, end_mlflow_run
    MLUTILS_AVAILABLE = True
except ImportError:
    MLUTILS_AVAILABLE = False

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Du mangler xgboost! Installer med: pip install xgboost")


from utils.telegram_utils import send_message

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_xgboost_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "best_xgboost_features.json")
LOG_PATH = os.path.join(MODEL_DIR, "train_log_xgboost.txt")
EARLYSTOP_ROUNDS = 10

def ensure_mlflow_run_closed():
    if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        print("[MLflow] Aktivt run fundet, lukker...")
        mlflow.end_run()

def log_to_file(line, prefix="[INFO] "):
    os.makedirs("logs", exist_ok=True)
    with open("logs/bot.log", "a", encoding="utf-8") as logf:
        logf.write(prefix + line)

def load_csv_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if str(first_line).startswith("#"):
        print("[INFO] Meta-header fundet i CSV – loader med skiprows=1")
        return pd.read_csv(file_path, skiprows=1)
    else:
        return pd.read_csv(file_path)

def train_xgboost_model(
    data_path,
    target_col="target",
    test_size=0.2,
    random_state=42,
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    early_stopping_rounds=EARLYSTOP_ROUNDS,
    verbose=True,
    save_model=True,
    use_mlflow=False,
    mlflow_exp="trading_ai"
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] Træning af XGBoost-model ({now})")
    print(f"[INFO] Læser data fra: {data_path}")

    df = load_csv_auto(data_path)
    assert target_col in df.columns, f"target_col '{target_col}' ikke fundet!"

    drop_cols = [target_col, "timestamp"] if "timestamp" in df.columns else [target_col]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]

    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        ignored = set(X.columns) - set(X_numeric.columns)
        print(f"[ADVARSEL] Ignorerer ikke-numeriske features: {ignored}")
    X = X_numeric

    # Gem brugte features
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(FEATURES_PATH, "w") as f:
            json.dump(list(X.columns), f, indent=2)
        print(f"[INFO] Gemte brugte features til: {FEATURES_PATH}")
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_artifact(FEATURES_PATH)

    # Split uden shuffle (tidsserie)
    df = df.reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")
    print("[INFO] Train slutter:", df.iloc[split_idx-1]["timestamp"] if "timestamp" in df.columns else split_idx-1)
    print("[INFO] Val starter:", df.iloc[split_idx]["timestamp"] if "timestamp" in df.columns else split_idx)
    print(f"[INFO] Unikke targets: {sorted(y.unique())}")

    # MLflow experiment tracking
    if use_mlflow and MLFLOW_AVAILABLE:
        ensure_mlflow_run_closed()
        if MLUTILS_AVAILABLE:
            setup_mlflow(experiment_name=mlflow_exp)
            start_mlflow_run(run_name=f"xgb_{now.replace(':','_')}")
        else:
            mlflow.set_experiment(mlflow_exp)
            mlflow.start_run(run_name=f"xgb_{now.replace(':','_')}")
        mlflow.log_params({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "test_size": test_size,
            "random_state": random_state,
            "early_stopping_rounds": early_stopping_rounds,
        })

    # Model (eval_metric KUN i constructor!)
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="logloss"  # KUN HER!
    )
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )

    # Evaluer
    y_pred = model.predict(X_val)
    acc = np.mean(y_pred == y_val)
    report = classification_report(y_val, y_pred, zero_division=0, output_dict=True)
    conf = confusion_matrix(y_val, y_pred)

    if verbose:
        print(f"[INFO] Val accuracy: {acc:.3f}")
        print(classification_report(y_val, y_pred, zero_division=0))
        print("Confusion matrix:\n", conf)

    # MLflow metrics
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metric("val_acc", acc)
        mlflow.log_metrics({f"f1_{k}": v["f1-score"] for k, v in report.items() if isinstance(v, dict) and "f1-score" in v})
        mlflow.log_param("best_iteration", getattr(model, "best_iteration", None))

    # Gem model (checkpoint)
    if save_model:
        try:
            model.save_model(MODEL_PATH)
            print(f"✅ Model gemt til: {MODEL_PATH}")
            if use_mlflow and MLFLOW_AVAILABLE:
                mlflow.xgboost.log_model(model, "model")
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke gemme model: {e}")

    # Log til fil
    log_str = f"\n== XGBoost-træningslog {now} ==\n" \
              f"Model: {MODEL_PATH}\nVal_acc: {acc:.3f}\nBest_iteration: {getattr(model, 'best_iteration', None)}\n" \
              f"Features: {list(X.columns)}\n\n" \
              f"Report:\n{classification_report(y_val, y_pred, zero_division=0)}\nConfusion:\n{conf}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_str)
    print(f"[INFO] Træningslog gemt til: {LOG_PATH}")

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_artifact(LOG_PATH)
        if MLUTILS_AVAILABLE:
            end_mlflow_run()
        else:
            mlflow.end_run()

    try:
        send_message(f"✅ XGBoost træning færdig – Val acc: {acc:.3f}")
    except Exception as e:
        print(f"[ADVARSEL] Telegram fejl: {e}")

    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Træn XGBoost-model til trading + MLflow logging + Early stopping")
    parser.add_argument("--data", type=str, required=True, help="Sti til features-data (.csv)")
    parser.add_argument("--target", type=str, default="target", help="Navn på target-kolonne")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split")
    parser.add_argument("--max_depth", type=int, default=6, help="Max tree depth")
    parser.add_argument("--n_estimators", type=int, default=100, help="Antal træer (estimators)")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio")
    parser.add_argument("--colsample_bytree", type=float, default=1.0, help="Colsample bytree")
    parser.add_argument("--early_stopping_rounds", type=int, default=EARLYSTOP_ROUNDS, help="Early stopping rounds")
    parser.add_argument("--no_save", action="store_true", help="Gem ikke model")
    parser.add_argument("--mlflow", action="store_true", help="Log til MLflow (experiment tracking)")
    parser.add_argument("--mlflow_exp", type=str, default="trading_ai", help="MLflow experiment name")
    args = parser.parse_args()

    train_xgboost_model(
        data_path=args.data,
        target_col=args.target,
        test_size=args.test_size,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=True,
        save_model=not args.no_save,
        use_mlflow=args.mlflow,
        mlflow_exp=args.mlflow_exp,
    )
