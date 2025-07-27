# scripts/finetune_winrate.py

import pandas as pd
import numpy as np
import argparse
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.project_path import PROJECT_ROOT
from utils.telegram_utils import send_telegram_message

def main():
    parser = argparse.ArgumentParser(description="Gridsearch/finetune targets og features for winrate.")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h_features.csv"), help="Input-feature-fil")
    parser.add_argument("--test_size", type=float, default=0.4, help="Test split")
    parser.add_argument("--min_winrate", type=float, default=0.55, help="Min winrate til alarm")
    parser.add_argument("--telegram", action="store_true", help="Send summary til Telegram")
    args = parser.parse_args()

    # === Features og targets du vil afprøve ===
    FEATURE_GROUPS = [
        ['close'],
        ['close', 'rsi_14'],
        ['close', 'rsi_14', 'ema_9'],
        ['close', 'rsi_14', 'ema_9', 'macd', 'macd_signal', 'vwap', 'atr_14'],
    ]
    TARGET_COLS = [c for c in pd.read_csv(args.input, nrows=1).columns if c.name.startswith("target_")]

    df = pd.read_csv(args.input)
    results = []

    for target_col in TARGET_COLS:
        df_valid = df.dropna(subset=[target_col])
        for feats in FEATURE_GROUPS:
            feats_exist = [f for f in feats if f in df_valid.columns]
            if len(feats_exist) != len(feats):
                continue
            X = df_valid[feats_exist]
            y = df_valid[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)
            model = lgb.LGBMClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            winrate = report['1']['recall'] if '1' in report else 0.0
            n_1 = np.sum(preds == 1)
            n_0 = np.sum(preds == 0)
            results.append({
                "target": target_col,
                "features": feats_exist,
                "accuracy": acc,
                "winrate": winrate,
                "n_pred_1": n_1,
                "n_pred_0": n_0,
                "n_test": len(y_test)
            })

    # Sortér og vis bedste resultater
    results = sorted(results, key=lambda x: x['winrate'], reverse=True)
    print("\n=== Top gridsearch resultater (winrate) ===")
    for res in results[:10]:
        print(f"{res['target']} {res['features']} -> Winrate: {res['winrate']:.2%}, Accuracy: {res['accuracy']:.2%}, n_1: {res['n_pred_1']}, n_0: {res['n_pred_0']} ({res['n_test']})")
    best = results[0]
    print("\n[INFO] Bedste konfiguration:")
    print(best)

    # Valgfrit: send Telegram-summary hvis ønsket
    if args.telegram:
        msg = (
            f"🚦 Gridsearch Winrate Summary\n"
            f"Bedste: {best['target']}\n"
            f"Features: {best['features']}\n"
            f"Winrate: {best['winrate']:.2%}\n"
            f"Accuracy: {best['accuracy']:.2%}\n"
            f"n_1: {best['n_pred_1']}, n_0: {best['n_pred_0']} (Test: {best['n_test']})"
        )
        if best['winrate'] >= args.min_winrate:
            msg = "✅ EDGE FUNDET!\n" + msg
        else:
            msg = "⚠️ EDGE IKKE FUNDENDE!\n" + msg
        send_telegram_message(msg)

if __name__ == "__main__":
    main()
