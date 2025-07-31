import pandas as pd
import ta  # pip install ta

from utils.project_path import PROJECT_ROOT


def create_features(input_csv, output_csv):
    # Prøv både komma og semikolon som separator!
    df = pd.read_csv(input_csv)
    if "close" not in df.columns:
        df = pd.read_csv(input_csv, sep=";")
        print("DEBUG: Forsøgte med sep=';'")
    print("DEBUG: Kolonner fundet:", list(df.columns))

    # Tving 'open', 'high', 'low', 'close', 'volume' til float
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Tjek om 'close' nu findes
    if "close" not in df.columns:
        print(
            "❌ FEJL: Kolonnen 'close' blev ikke fundet i CSV. Kolonner:",
            list(df.columns),
        )
        print("Eksempel på første række:", df.iloc[0].to_dict())
        raise KeyError("CSV-filen mangler 'close'-kolonne. Tjek separator og header!")

    # Feature engineering
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()

    df.dropna(inplace=True)  # Fjern NaN fra startperioder
    df.to_csv(output_csv, index=False)
    print(f"✅ Features gemt: {output_csv}")


if __name__ == "__main__":
    # AUTO PATH CONVERTED
    create_features(
        PROJECT_ROOT / "data" / "BTCUSDT_1h.csv",
        PROJECT_ROOT / "data" / "BTCUSDT_1h_features.csv",
    )
