import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import datetime
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import argparse
import time
import numpy as np
import shap

from sklearn.inspection import permutation_importance

from utils.robust_utils import safe_run
from utils.telegram_utils import send_image, send_message
from utils.feature_logging import (
    log_top_features_to_md,
    log_top_features_csv,
    send_top_features_telegram,
)

# ---- NY: Robust features-loader ----
def read_features_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        print("🔎 Meta-header fundet – springer første linje over (skiprows=1).")
        df = pd.read_csv(file_path, skiprows=1)
    else:
        df = pd.read_csv(file_path)
    print("🔎 Kolonner i indlæst features-DF:", list(df.columns))
    return df

def get_random_seed(cli_seed=None):
    if cli_seed is not None:
        return int(cli_seed)
    return int(time.time()) % 100000

def to_str(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return str(x.item())
        else:
            return str(x.tolist())
    return str(x)

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except:
        return "unknown"

def log_model_metrics(filename, y_true, y_pred, model_name, version="v1", random_seed=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()

    row = {
        "timestamp": timestamp,
        "version": version,
        "git_hash": git_hash,
        "model": model_name,
        "random_seed": random_seed,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    df = pd.DataFrame([row])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

    cm_img_file = filename.replace('.csv', f'_confmat_{timestamp.replace(" ","_").replace(":","-")}.png')
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix\n{model_name} {timestamp}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_img_file)
    plt.close()
    print(f"✅ Confusion matrix gemt som billede: {cm_img_file}")

    cm_file = filename.replace('.csv', f'_confmat_{timestamp.replace(" ","_").replace(":","-")}.csv')
    cm_df = pd.DataFrame(cm)
    cm_df["timestamp"] = timestamp
    cm_df.to_csv(cm_file, index=False)
    print(f"✅ Confusion matrix gemt som CSV: {cm_file}")

def load_best_accuracy(meta_path="models/best_model_meta.json"):
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("accuracy", 0.0)
    return 0.0

def load_best_model_features(meta_path="models/best_model_meta.json"):
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("features", None)
    return None

def save_best_model(model, accuracy, model_path="models/best_model.pkl", meta_path="models/best_model_meta.json", features=None, random_seed=None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        "accuracy": accuracy,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "random_seed": random_seed
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Ny bedste model gemt: {model_path} (accuracy: {accuracy:.4f}) med {len(features) if features else 'ukendt'} features")

def plot_feature_importance(feature_names, importance_scores, out_path="outputs/feature_importance.png", method="Permutation", top_n=15):
    feature_names = np.array(feature_names)
    importance_scores = np.array(importance_scores)
    idx = np.argsort(importance_scores)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[idx], importance_scores[idx])
    plt.xlabel("Feature importance")
    plt.title(f"{method} Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def calculate_permutation_importance(model, X_val, y_val):
    X_val = X_val.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number]).fillna(0)
    X_val = X_val.astype("float64")
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
    importance_scores = np.atleast_1d(result.importances_mean)
    feature_names = np.array(X_val.columns)
    assert feature_names.shape[0] == importance_scores.shape[0], "Mismatch mellem features og importance_scores"
    sorted_idx = importance_scores.argsort()[::-1]
    return [to_str(x) for x in feature_names[sorted_idx]], [float(x) for x in importance_scores[sorted_idx]]

def calculate_shap_importance(model, X_val):
    X_val = X_val.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number]).fillna(0)
    X_val = X_val.astype("float64")
    explainer = shap.Explainer(model, X_val)
    shap_values = explainer(X_val, check_additivity=False)
    values = np.atleast_2d(shap_values.values)
    try:
        values = values.astype(np.float64)
    except Exception as e:
        raise ValueError(f"Kan ikke konvertere SHAP values til float64! Fejl: {e}\nSample: {values[:2]}")
    assert np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer), \
        f"SHAP values er ikke numeriske! dtype: {values.dtype}"
    if values.ndim == 3:
        shap_imp = np.abs(values).mean(axis=(0, 2))
    else:
        shap_imp = np.abs(values).mean(axis=0)
    feature_names = np.array(X_val.columns)
    assert feature_names.shape[0] == shap_imp.shape[0], f"Mismatch mellem features og SHAP importance: {feature_names.shape[0]} vs {shap_imp.shape[0]}"
    sorted_idx = shap_imp.argsort()[::-1]
    return [to_str(x) for x in feature_names[sorted_idx]], [float(x) for x in shap_imp[sorted_idx]]

def plot_shap_importance(feature_names, shap_scores, out_path="outputs/shap.png", top_n=15):
    feature_names = np.array(feature_names)
    shap_scores = np.array(shap_scores, dtype=np.float64)
    idx = np.argsort(shap_scores)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[idx], shap_scores[idx])
    plt.xlabel("SHAP Importance")
    plt.title("SHAP Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def auto_feature_selection(model, X_train, y_train, X_val, y_val, threshold=0.01, min_features=5):
    X_train = X_train.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number]).fillna(0)
    X_train = X_train.astype("float64")
    X_val = X_val.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number]).fillna(0)
    X_val = X_val.astype("float64")
    names, scores = calculate_permutation_importance(model, X_val, y_val)
    keep_idx = [s >= threshold for s in scores]
    if sum(keep_idx) < min_features:
        keep_idx = [True if i < min_features else k for i, k in enumerate(keep_idx)]
    selected_features = [n for n, k in zip(names, keep_idx) if k]
    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X_train_sel, y_train)
    return new_model, selected_features

def run_feature_importance_and_selection(
    model, X_train, y_train, X_val, y_val, strategy_name="ML", telegram_chat_id=None
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    outdir = "outputs/"
    os.makedirs(outdir, exist_ok=True)

    perm_png = f"{outdir}feature_importance_perm_{strategy_name}_{timestamp}.png"
    feat_names, imp_scores = calculate_permutation_importance(model, X_val, y_val)
    plot_feature_importance(feat_names, imp_scores, out_path=perm_png, method="Permutation", top_n=15)
    top_perm = list(zip(feat_names[:5], imp_scores[:5]))

    shap_png = f"{outdir}feature_importance_shap_{strategy_name}_{timestamp}.png"
    shap_names, shap_scores = calculate_shap_importance(model, X_val)
    plot_shap_importance(shap_names, shap_scores, out_path=shap_png, top_n=15)
    top_shap = list(zip(shap_names[:5], shap_scores[:5]))

    new_model, selected_features = auto_feature_selection(model, X_train, y_train, X_val, y_val, threshold=0.01)

    log_path = f"{outdir}feature_importance_{strategy_name}_{timestamp}.csv"
    shap_dict = dict(zip(shap_names, shap_scores))
    shap_importance_col = [shap_dict.get(name, np.nan) for name in feat_names]
    df_log = pd.DataFrame({
        "feature": feat_names,
        "permutation_importance": imp_scores,
        "shap_importance": shap_importance_col
    })
    df_log.to_csv(log_path, index=False)

    log_top_features_to_md(top_perm, md_path="BotStatus.md", model_name=strategy_name)
    log_top_features_csv(top_perm, csv_path="data/top_features_history.csv", model_name=strategy_name)

    perm_caption = f"📊 Permutation Feature Importance ({strategy_name})\nTop-5:\n" + \
                   "\n".join([f"{n}: {s:.3f}" for n, s in top_perm])
    shap_caption = f"🌈 SHAP Feature Importance ({strategy_name})\nTop-5:\n" + \
                   "\n".join([f"{n}: {s:.3f}" for n, s in top_shap])
    if telegram_chat_id:
        send_image(perm_png, caption=perm_caption, chat_id=telegram_chat_id)
        send_image(shap_png, caption=shap_caption, chat_id=telegram_chat_id)
        summary = (
            f"✅ Automatisk Feature Importance ({strategy_name})\n"
            f"Top-5 (Permutation): {top_perm}\n"
            f"Top-5 (SHAP): {top_shap}\n"
            f"Udvalgte features til retræning: {list(selected_features)}"
        )
        send_message(summary, chat_id=telegram_chat_id)
        send_top_features_telegram(top_perm, send_message, chat_id=telegram_chat_id, model_name=strategy_name)

    print("Feature importance summary:")
    print(perm_caption)
    print(shap_caption)
    print(f"Udvalgte features til retræning: {list(selected_features)}")

    return new_model, selected_features, perm_png, shap_png, log_path

def train_model(features, target_col='target', version="v1", telegram_chat_id=None, random_seed=None):
    if isinstance(features, str):
        df = read_features_auto(features)
    else:
        df = features.copy()

    # --- DEBUG: Print kolonner ---
    print("🔎 Kolonner i trænings-DF:", list(df.columns))

    drop_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col != target_col]
    feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]
    X = df[feature_cols]
    y = df[target_col]

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    X = X.astype("float64")
    print("Dtypes (features):", X.dtypes.value_counts())
    not_float_cols = X.dtypes[X.dtypes != "float64"]
    if not_float_cols.shape[0] > 0:
        print("⚠️ ADVARSEL: Nogle features er ikke float64:", not_float_cols)
    if X.isnull().any().any():
        print("⚠️ ADVARSEL: Der er NaN i features!", list(X.columns[X.isnull().any()]))

    if random_seed is None:
        random_seed = get_random_seed()
    print(f"🎲 RANDOM_SEED brugt for split og model: {random_seed}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    log_model_metrics("data/model_eval.csv", y_test, preds, "RandomForest", version=version, random_seed=random_seed)

    new_model, selected_features, *_ = run_feature_importance_and_selection(
        model, X_train, y_train, X_test, y_test, strategy_name="ML", telegram_chat_id=telegram_chat_id
    )

    best_acc = load_best_accuracy()
    model_path = "models/best_model.pkl"
    meta_path = "models/best_model_meta.json"

    if acc > best_acc or not os.path.exists(model_path):
        save_best_model(new_model, acc, model_path, meta_path, features=selected_features, random_seed=random_seed)
    else:
        print("ℹ️ Model ikke gemt – accuracy er ikke bedre end tidligere.")

    return new_model, model_path, selected_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed til split/model (default=random hver gang)")
    parser.add_argument("--features", type=str, default=None, help="Path til features-CSV (finder selv nyeste hvis tom)")
    args = parser.parse_args()

    # ---- Find altid nyeste feature-CSV hvis ikke angivet! ----
    features_path = args.features
    if not features_path or not os.path.exists(features_path):
        print(f"➡️ Finder nyeste features-CSV i outputs/feature_data/")
        import glob
        feature_files = sorted(glob.glob("outputs/feature_data/btcusdt_1h_features*.csv"))
        if not feature_files:
            raise FileNotFoundError("Ingen features-CSV fundet i outputs/feature_data/")
        features_path = feature_files[-1]
        print(f"➡️ Bruger: {features_path}")

    os.makedirs("models", exist_ok=True)
    telegram_chat_id = None
    model, model_path, selected_features = train_model(
        features_path,
        version="v1.0.1",
        telegram_chat_id=telegram_chat_id,
        random_seed=args.seed
    )
    print("Trænede på features:", selected_features)
    print(f"🎲 RANDOM_SEED brugt: {args.seed}")

if __name__ == "__main__":
    safe_run(main)
