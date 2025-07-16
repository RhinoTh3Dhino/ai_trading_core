# ml/train_ml_model.py

import os
import json
import pickle
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Sti til feature-CSV")
    parser.add_argument("--target", type=str, default="target", help="Target-kolonne")
    parser.add_argument("--model_out", type=str, default="models/best_ml_model.pkl")
    parser.add_argument("--features_out", type=str, default="models/best_ml_features.json")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.features_out), exist_ok=True)

    df = pd.read_csv(args.data, comment="#").dropna(subset=[args.target])
    num_cols = df.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in num_cols if c not in [args.target, "future_return"]]
    # Ekstra: advarsel hvis nogen kolonner mangler fra tidligere feature set
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"‼️ ADVARSEL: Følgende features manglede i data og blev tilføjet med 0: {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[feature_cols]
    y = df[args.target].astype(int)

    # Split 70/15/15 (kun for val-test, retrain på alt bagefter)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    print("\nVal Accuracy:", accuracy_score(y_val, val_preds))
    print(classification_report(y_val, val_preds))

    # Test på test-sæt (valgfrit)
    test_preds = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds))

    # Endelig træning på alt data!
    clf.fit(X, y)
    with open(args.model_out, "wb") as f:
        pickle.dump(clf, f)
    with open(args.features_out, "w") as f:
        json.dump(feature_cols, f)
    print(f"[INFO] ML-model og feature-liste gemt til '{args.model_out}', '{args.features_out}'")
    print(f"[INFO] Features: {feature_cols}")

    # Test-load (robusthedstjek)
    with open(args.model_out, "rb") as f:
        loaded_model = pickle.load(f)
    assert hasattr(loaded_model, "predict"), "❌ Model har ikke predict()-metode!"
    print("✅ Test-load af ML-model: OK")

if __name__ == "__main__":
    main()
