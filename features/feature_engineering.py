from utils.project_path import PROJECT_ROOT  # AUTO PATH CONVERTED
# features/feature_engineering.py

import pandas as pd
import hashlib
import argparse
import os
import glob
from datetime import datetime

# --- Versionsinfo fra versions.py ---
try:
    from versions import (
        PIPELINE_VERSION, PIPELINE_COMMIT,
        FEATURE_VERSION, ENGINE_VERSION, ENGINE_COMMIT, MODEL_VERSION, LABEL_STRATEGY
    )
except ImportError:
    PIPELINE_VERSION = PIPELINE_COMMIT = FEATURE_VERSION = ENGINE_VERSION = ENGINE_COMMIT = MODEL_VERSION = LABEL_STRATEGY = "unknown"

# === Hvis der findes en features-liste fra tidligere model, loades denne ===
LSTM_FEATURES_PATH = PROJECT_ROOT / "models" / "lstm_features.csv"  # AUTO PATH CONVERTED

def feature_hash(df: pd.DataFrame) -> str:
    """Returnerer hash af DataFrame for versionering."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    import ta
    df = df.copy()
    # Sikrer korrekte datatyper for alle kernekolonner
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Klassiske tekniske indikatorer ---
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # --- Ekstra: glidende gennemsnit, volatilitet, returns ---
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()
    df["volatility_21"] = df["close"].rolling(window=21).std()
    df["returns_1h"] = df["close"].pct_change(periods=1)
    df["returns_24h"] = df["close"].pct_change(periods=24)

    # --- Regime: bull=1, bear=-1, neutral=0 (kan bruges direkte i ML) ---
    df["regime"] = 0
    df.loc[df["ema_50"] > df["ema_200"], "regime"] = 1
    df.loc[df["ema_50"] < df["ema_200"], "regime"] = -1

    # --- Target engineering ---
    df["future_return"] = df["close"].shift(-24) / df["close"] - 1
    df["target"] = (df["future_return"] > 0.01).astype(int)

    # --- Drop rækker med NaN i kritiske kolonner ---
    critical_cols = [
        "close", "ema_200", "rsi_14", "ema_9", "ema_21", "ema_50",
        "macd", "macd_signal", "atr_14", "regime", "target"
    ]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    after = len(df)
    if before != after:
        print(f"ℹ️ {before-after} rækker droppet pga. NaN i features/target.")

    # --- Sikrer at alle tidligere brugte features (lstm_features.csv) er med ---
    if os.path.exists(LSTM_FEATURES_PATH):
        lstm_features = pd.read_csv(LSTM_FEATURES_PATH, header=None)[0].tolist()
        missing = [col for col in lstm_features if col not in df.columns]
        if missing:
            print(f"‼️ ADVARSEL: Følgende features mangler og bliver fyldt med 0: {missing}")
            for col in missing:
                df[col] = 0.0  # Udfyld manglende features med 0 (alternativ: np.nan eller fill f.eks. med mean)
        # Sortér kolonner så rækkefølge matcher lstm_features
        df = df[[col for col in lstm_features if col in df.columns] + [c for c in df.columns if c not in lstm_features]]
    return df

def find_latest_datafile(pattern: str = PROJECT_ROOT / "data" / "BTCUSDT_1h_*.csv"  # AUTO PATH CONVERTED) -> str:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        files = sorted(glob.glob(PROJECT_ROOT / "data" / "*.csv"  # AUTO PATH CONVERTED), key=os.path.getmtime)
    files = [f for f in files if all(x not in os.path.basename(f).lower() for x in ['feature', 'history', 'importance', 'result'])]
    return files[-1] if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Input-CSV med rå data (finder selv nyeste hvis ikke angivet)")
    parser.add_argument("--output", type=str, required=True, help="Output-CSV med features")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Feature-version (logges i meta og header)")
    args = parser.parse_args()

    input_path = args.input
    if not input_path or not os.path.exists(input_path):
        print(f"⚠️ Inputfil ikke angivet eller ikke fundet. Søger efter nyeste datafil i 'data/'...")
        latest = find_latest_datafile()
        if latest:
            print(f"➡️  Bruger nyeste datafil: {latest}")
            input_path = latest
        else:
            print(f"❌ Ingen datafil fundet i 'data/' mappen. Stopper.")
            return

    df = pd.read_csv(input_path)
    print("🔎 Kolonner i rå data:", list(df.columns))
    if "close" not in df.columns:
        print(f"❌ Inputfilen '{input_path}' mangler kolonnen 'close'. Tjek din rå data!")
        return

    df_feat = add_features(df)
    print("🔎 Kolonner EFTER feature engineering:", list(df_feat.columns))
    if "regime" not in df_feat.columns:
        print("❌ FEJL: 'regime' blev ikke tilføjet i feature engineering!")
        print(df_feat.head())
        return
    if "target" not in df_feat.columns:
        print("❌ FEJL: 'target' blev ikke tilføjet i feature engineering!")
        print(df_feat.head())
        return

    hash_val = feature_hash(df_feat)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Gem features med meta-header (inkl. versionsinfo)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(
            f"# Feature version: {args.version} | hash: {hash_val} | generated: {timestamp} | "
            f"pipeline_commit: {PIPELINE_COMMIT} | engine_commit: {ENGINE_COMMIT}\n"
        )
        df_feat.to_csv(f, index=False)

    # Gem hash/version i separat meta-fil
    meta_path = os.path.splitext(args.output)[0] + "_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"feature_version: {args.version}\n")
        f.write(f"feature_hash: {hash_val}\n")
        f.write(f"generated: {timestamp}\n")
        f.write(f"input_file: {input_path}\n")
        f.write(f"pipeline_version: {PIPELINE_VERSION}\n")
        f.write(f"pipeline_commit: {PIPELINE_COMMIT}\n")
        f.write(f"engine_version: {ENGINE_VERSION}\n")
        f.write(f"engine_commit: {ENGINE_COMMIT}\n")
        f.write(f"feature_version_code: {FEATURE_VERSION}\n")
        f.write(f"model_version: {MODEL_VERSION}\n")
        f.write(f"label_strategy: {LABEL_STRATEGY}\n")

    print(f"✅ Features gemt til: {args.output} (hash: {hash_val})")

if __name__ == "__main__":
    main()