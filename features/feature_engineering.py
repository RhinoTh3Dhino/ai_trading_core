import pandas as pd
import ta
import glob
import os

def add_features(df):
    df = df.copy()
    df["close"] = pd.to_numeric(df["close"])
    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["volume"] = pd.to_numeric(df["volume"])

    # Tekniske indikatorer
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close=df["close"], window=200).ema_indicator()
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # Enkel regime (kan udvides)
    df["regime"] = "neutral"
    df.loc[df["ema_50"] > df["ema_200"], "regime"] = "bull"
    df.loc[df["ema_50"] < df["ema_200"], "regime"] = "bear"

    # Fyld evt. NaN op med 0 eller forrige værdi
    df = df.fillna(method="bfill").fillna(0)
    return df

def add_target(df, threshold=0.002):
    """
    Tilføjer en target-kolonne for supervised learning.
    1 = prisen stiger mere end threshold (0.2%)
   -1 = prisen falder mere end threshold
    0 = ellers (ingen stærk bevægelse)
    """
    df = df.copy()
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target"] = 0
    df.loc[df["future_return"] > threshold, "target"] = 1
    df.loc[df["future_return"] < -threshold, "target"] = -1
    df.drop(columns=["future_return"], inplace=True)
    return df

if __name__ == "__main__":
    # Automatisk find nyeste datafil
    files = glob.glob("outputs/feature_data/btc_1h_features_*.csv")
    if not files:
        print("❌ Ingen datafiler fundet i outputs/feature_data/")
        exit(1)
    path = max(files, key=os.path.getctime)
    print(f"🔄 Opdaterer features og target i: {path}")

    df = pd.read_csv(path)
    df = add_features(df)
    df = add_target(df, threshold=0.002)
    df.to_csv(path, index=False)
    print(f"✅ Features og target tilføjet og data opdateret: {path}")
    print(f"Kolonner nu: {list(df.columns)} | Rækker: {len(df)}")
