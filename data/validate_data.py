import pandas as pd

from utils.project_path import PROJECT_ROOT


def validate_csv_data(filepath):
    """
    Validerer indholdet af en trading-csv og udskriver info/statistik.
    Tjekker kolonner, datatyper, NaN-værdier, basic statistik.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Kunne ikke indlæse filen: {e}")
        return False

    print(f"✅ Data indlæst: {filepath}")
    print(f"Kolonner: {list(df.columns)}")
    print(df.info())
    print(df.head(5))

    # Tjek for manglende værdier
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️ Manglende værdier pr. kolonne:")
        print(missing)
    else:
        print("✅ Ingen manglende værdier fundet.")

    # Tjek datatyper
    print("\nDatatyper:")
    print(df.dtypes)

    # Tjek basic statistik på pris/volume
    if "close" in df.columns:
        print("\nStatistik på 'close':")
        print(df["close"].describe())

    return True


if __name__ == "__main__":
    # Eksempel på brug
    # AUTO PATH CONVERTED
    validate_csv_data(PROJECT_ROOT / "data" / "BTCUSDT_1h.csv")
