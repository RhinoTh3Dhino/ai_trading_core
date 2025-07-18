# tests/test_model_predict.py
"""
Test af ML-model load + predict + robust fallback.
Kør fra projektroden:  python tests/test_model_predict.py
"""

import os
import sys
import pandas as pd

# Gør det muligt at importere fra bot/ og utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.engine import load_ml_model, reconcile_features

# ---- KONFIG ----
FEATURE_PATH = "outputs/feature_data/btcusdt_1h_features_v1.0_20250718.csv"  # <-- Ret til din nyeste feature-fil!
N_ROWS = 20  # Hvor mange rækker vi vil teste på

def main():
    # 1. Indlæs features
    if not os.path.exists(FEATURE_PATH):
        print(f"❌ Feature-fil ikke fundet: {FEATURE_PATH}")
        return
    df = pd.read_csv(FEATURE_PATH)
    print(f"✅ Indlæst {len(df)} rækker fra {FEATURE_PATH}")
    df = df.head(N_ROWS)
    
    # 2. Prøv at loade ML-model og feature-liste
    ml_model, ml_features = load_ml_model()
    if ml_model is not None and ml_features is not None:
        print("✅ ML-model og feature-liste fundet!")
        # 3. Matcher features og predict
        X = reconcile_features(df, ml_features)
        try:
            preds = ml_model.predict(X)
            print(f"✅ Model predict OK! Første 10 signaler: {preds[:10]}")
        except Exception as e:
            print(f"❌ FEJL ved predict(): {e}")
    else:
        print("⚠️ ML-model/feature-liste ikke fundet. Tester fallback...")
        # 4. Fallback – lav random signaler (systemet skal ikke crashe!)
        df["signal"] = [1 if i % 2 == 0 else -1 for i in range(len(df))]
        print(f"✅ Random signaler genereret: {df['signal'].tolist()[:10]}")

if __name__ == "__main__":
    main()
