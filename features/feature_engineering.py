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

def feature_hash(df: pd.DataFrame) -> str:
    """Returnerer hash af DataFrame for versionering."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    import ta
    df = df.copy()
    # Sikrer korrekte datatyper
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    # Robust regime-beregning (bull: 0, bear: 1)
    df["regime"] = 0
    df.loc[df["ema_50"] > df["ema_200"], "regime"] = 0  # bull
    df.loc[df["ema_50"] < df["ema_200"], "regime"] = 1  # bear
    # Target-label: 1 hvis næste close > nuværende close, ellers 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ---- Drop rækker med NaN i kritiske kolonner ----
    critical_cols = [
        "close", "ema_200", "rsi_14", "ema_9", "ema_21", "ema_50",
        "macd", "macd_signal", "atr_14", "regime", "target"
    ]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    after = len(df)
    if before != after:
        print(f"ℹ️ {before-after} rækker droppet pga. NaN i features/target.")
    return df

def find_latest_datafile(pattern: str = "data/BTCUSDT_1h_*.csv") -> str:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        files = sorted(glob.glob("data/*.csv"), key=os.path.getmtime)
    files = [f for f in files if all(x not in os.path.basename(f).lower() for x in ['feature', 'history', 'importance', 'result'])]
    return files[-1] if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Input-CSV med rå data (valgfri, finder selv nyeste hvis ikke angivet)")
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
    if "close" not in df.columns:
        print(f"❌ Inputfilen '{input_path}' mangler kolonnen 'close'. Tjek din rå data!")
        return

    df_feat = add_features(df)
    if "regime" not in df_feat.columns:
        print("❌ FEJL: 'regime' blev ikke tilføjet i feature engineering!")
        return
    if "target" not in df_feat.columns:
        print("❌ FEJL: 'target' blev ikke tilføjet i feature engineering!")
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

    # Gem hash/version i separat meta-fil (inkl. versionsinfo)
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
