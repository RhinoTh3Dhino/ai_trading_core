import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import pandas as pd
import numpy as np
from features.preprocessing import clean_dataframe, normalize_zscore


def test_clean_and_normalize():
    # Dummy data med NaN, inf og outlier
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, np.nan, 5000, 103, 105, np.inf, 104],
            "volume": [10, 12, np.nan, 13, 11, 12, 999, 10, np.inf],
        }
    )

    # Data cleaning med lavere z-score threshold for små datasæt
    df_clean = clean_dataframe(df.copy(), outlier_z=1.5)
    print("\n[EFTER CLEANING]\n", df_clean)

    assert (
        not df_clean.isna().any().any()
    ), f"Der er stadig NaN efter cleaning!\n{df_clean}"
    assert not np.isinf(
        df_clean.values
    ).any(), f"Der er stadig inf efter cleaning!\n{df_clean}"
    assert (
        df_clean["close"].max() < 1000
    ), f"Outlier i 'close' er ikke fjernet!\n{df_clean}"

    # Normalisering
    df_z = normalize_zscore(df_clean.copy(), columns=["close", "volume"])
    print("\n[EFTER NORMALISERING]\n", df_z)

    assert "close_z" in df_z.columns, "Kolonne close_z mangler!"
    assert "volume_z" in df_z.columns, "Kolonne volume_z mangler!"
    assert (
        abs(df_z["close_z"].mean()) < 1e-6
    ), f"Mean for close_z er ikke ~0: {df_z['close_z'].mean()}"
    assert (
        abs(df_z["volume_z"].mean()) < 1e-6
    ), f"Mean for volume_z er ikke ~0: {df_z['volume_z'].mean()}"


if __name__ == "__main__":
    test_clean_and_normalize()
    print("✅ Data cleaning og normalisering virker!")
