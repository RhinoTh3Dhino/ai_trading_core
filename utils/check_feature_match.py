# utils/check_feature_match.py
import os
import json
import pandas as pd
import argparse

MODEL_DIR = "models"
PYTORCH_FEATURES_PATH = os.path.join(MODEL_DIR, "best_pytorch_features.json")
LSTM_FEATURES_PATH = os.path.join(MODEL_DIR, "lstm_features.csv")

def load_trained_features():
    """Finder og loader feature-listen for PyTorch eller LSTM model."""
    if os.path.exists(PYTORCH_FEATURES_PATH):
        with open(PYTORCH_FEATURES_PATH, "r") as f:
            features = json.load(f)
        print(f"[INFO] PyTorch features: {features}")
        return features
    elif os.path.exists(LSTM_FEATURES_PATH):
        features = pd.read_csv(LSTM_FEATURES_PATH, header=None)[0].tolist()
        print(f"[INFO] LSTM features: {features}")
        return features
    else:
        print("❌ Ingen features gemt for model! Træn model først.")
        return None

def check_feature_match(feature_file, verbose=True):
    features_trained = load_trained_features()
    if features_trained is None:
        return False
    # Indlæs CSV med skiprows hvis meta-header
    with open(feature_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        df = pd.read_csv(feature_file, skiprows=1)
    else:
        df = pd.read_csv(feature_file)
    features_actual = list(df.columns)
    # Fjern ikke-feature kolonner
    ignore_cols = set(["timestamp", "future_return", "target", "signal"])
    features_actual = [col for col in features_actual if col not in ignore_cols]
    # Sammenlign
    missing = [f for f in features_trained if f not in features_actual]
    extra   = [f for f in features_actual if f not in features_trained]
    order_mismatch = features_trained != [col for col in features_actual if col in features_trained]

    if verbose:
        print("\n=== Feature Match Report ===")
        print("Model-features: ", features_trained)
        print("CSV-features:   ", features_actual)
        if missing:
            print(f"‼️ Mangler features i CSV: {missing}")
        if extra:
            print(f"⚠️ Ekstra features i CSV (ikke brugt i model): {extra}")
        if order_mismatch:
            print(f"🔄 Feature-rækkefølge stemmer ikke 100%!")
        if not missing and not order_mismatch and not extra:
            print("✅ Features i CSV matcher 100% modellen (navne, rækkefølge, antal).")

    return not missing and not order_mismatch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tjek at feature-CSV matcher din models feature-liste (rækkefølge og navne)")
    parser.add_argument("--csv", type=str, required=True, help="Path til din feature-CSV (med eller uden meta-header)")
    args = parser.parse_args()
    ok = check_feature_match(args.csv)
    if ok:
        print("✅ CHECK OK: CSV er klar til inference med din nuværende DL-model.")
    else:
        print("❌ FEJL: Justér features eller retræn model så de matcher præcis.")
