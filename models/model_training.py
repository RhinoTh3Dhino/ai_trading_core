import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime
import os
import json
import joblib

def log_model_metrics(filename, accuracy, report, confusion):
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {'label': label}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['accuracy'] = accuracy

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)

    cm_file = filename.replace('.csv', '_confmat.csv')
    cm_df = pd.DataFrame(confusion)
    cm_df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(cm_file):
        cm_df.to_csv(cm_file, index=False)
    else:
        cm_df.to_csv(cm_file, mode='a', index=False, header=False)

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

def train_model(features_csv):
    df = pd.read_csv(features_csv)

    drop_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) or col == 'target']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    confmat = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    log_model_metrics("data/model_eval.csv", acc, report, confmat)

    best_acc = load_best_accuracy()
    model_path = "models/best_model.pkl"
    meta_path = "models/best_model_meta.json"

    if acc > best_acc or not os.path.exists(model_path):
        save_best_model(model, acc, model_path, meta_path)
    else:
        print("ℹ️ Model ikke gemt – accuracy er ikke bedre end tidligere.")

    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model("data/BTCUSDT_1h_features.csv")
