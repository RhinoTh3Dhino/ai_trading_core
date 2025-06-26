import pandas as pd
import hashlib
import argparse
import os
import glob

def feature_hash(df: pd.DataFrame) -> str:
    """Returnerer hash af DataFrame for versionering."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Her kan du samle alle dine feature-beregninger
    # Eksempel: (tilpas med dine egne indikatorer)
    import ta
    df = df.copy()
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

def find_latest_datafile(pattern: str = "data/BTCUSDT_1h_*.csv") -> str:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        # Fallback: find any CSV file in data/
        files = sorted(glob.glob("data/*.csv"), key=os.path.getmtime)
    return files[-1] if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Input-CSV med rå data (valgfri, finder selv nyeste hvis ikke angivet)")
    parser.add_argument("--output", type=str, required=True, help="Output-CSV med features")
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
    df_feat = add_features(df)
    hash_val = feature_hash(df_feat)
    df_feat.to_csv(args.output, index=False)
    # Gem hash/version i separat fil
    meta_path = os.path.splitext(args.output)[0] + "_meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"feature_hash: {hash_val}\n")
    print(f"✅ Features gemt til: {args.output} (hash: {hash_val})")

if __name__ == "__main__":
    main()
