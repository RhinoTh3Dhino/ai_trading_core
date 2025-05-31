import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime
import os

def log_model_metrics(filename, accuracy, report, confusion):
    """
    Logger modelmetrics og confusion-matrix til CSV, inkl. timestamp og accuracy.
    """
    # Konverter classification_report (dict) til dataframe
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {'label': label}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['accuracy'] = accuracy

    # Gem til CSV (append eller opret ny)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)

    # Confusion matrix til separat fil
    cm_file = filename.replace('.csv', '_confmat.csv')
    cm_df = pd.DataFrame(confusion)
    cm_df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(cm_file):
        cm_df.to_csv(cm_file, index=False)
    else:
        cm_df.to_csv(cm_file, mode='a', index=False, header=False)

def train_model(features_csv):
    """
    Træner en RandomForest på feature-CSV, logger eval-metrics automatisk.
    """
    df = pd.read_csv(features_csv)

    # --- Ekskluder ikke-numeriske kolonner (fx 'datetime', 'symbol', mm.) ---
    drop_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) or col == 'target']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df['target']

    # --- Split og træn ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluer ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    confmat = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # --- Automatisk logging ---
    log_model_metrics("data/model_eval.csv", acc, report, confmat)

    # # Gem model hvis ønsket
    # import joblib; joblib.dump(model, 'models/baseline_rf.pkl')
    return model

if __name__ == "__main__":
    train_model("data/BTCUSDT_1h_features.csv")
