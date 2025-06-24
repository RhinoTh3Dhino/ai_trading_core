import pandas as pd
import ta

def add_features(df):
    df = df.copy()
    df["close"] = pd.to_numeric(df["close"])
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["ema_200"] = ta.trend.EMAIndicator(close=df["close"], window=200).ema_indicator()
    
    # Enkel regime-mapping (kan udvides senere)
    df["regime"] = (df["close"].diff(200) > 0).astype(int)
    df["regime"] = df["regime"].map({1: "bull", 0: "bear"})
    df["regime"] = df["regime"].fillna("neutral")
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
    path = "outputs/feature_data/btc_1h_features_20250624.csv"
    df = pd.read_csv(path)
    df = add_features(df)
    df = add_target(df, threshold=0.002)
    df.to_csv(path, index=False)
    print("✅ Features og target tilføjet og data opdateret:", path)
