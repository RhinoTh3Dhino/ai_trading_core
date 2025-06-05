import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import datetime
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except:
        return "unknown"

def log_model_metrics(filename, y_true, y_pred, model_name, version="v1"):
    # Beregn metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()

    # Saml i DataFrame
    row = {
        "timestamp": timestamp,
        "version": version,
        "git_hash": git_hash,
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    df = pd.DataFrame([row])

    # Gem eller tilføj til CSV
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

    # Gem confusion matrix som billede (.png)
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

    # Gem confusion matrix også som CSV (valgfrit)
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

def save_best_model(model, accuracy, model_path="models/best_model.pkl", meta_path="models/best_model_meta.json"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        "accuracy": accuracy,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Ny bedste model gemt: {model_path} (accuracy: {accuracy:.4f})")

def train_model(features, target_col='target', version="v1"):
    """
    Træner en RandomForest-model på features (DataFrame eller CSV-fil).
    Returnerer (model, model_path, feature_cols)
    """
    if isinstance(features, str):
        df = pd.read_csv(features)
    else:
        df = features.copy()

    # Fjern ikke-numeriske kolonner undtagen target
    drop_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col != target_col]
    feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # Log metrics og confusion matrix
    log_model_metrics("data/model_eval.csv", y_test, preds, "RandomForest", version=version)

    best_acc = load_best_accuracy()
    model_path = "models/best_model.pkl"
    meta_path = "models/best_model_meta.json"

    if acc > best_acc or not os.path.exists(model_path):
        save_best_model(model, acc, model_path, meta_path)
    else:
        print("ℹ️ Model ikke gemt – accuracy er ikke bedre end tidligere.")

    return model, model_path, feature_cols

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    # CLI-test
    model, model_path, feature_cols = train_model("data/BTCUSDT_1h_features.csv", version="v1.0.1")
    print("Trænede på features:", feature_cols)
