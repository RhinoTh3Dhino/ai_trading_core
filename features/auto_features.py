# features/auto_features.py

from pathlib import Path
from typing import Union

import pandas as pd

from features.features_pipeline import generate_features

RAW_DIR = Path("outputs/data")


def _generate_from_raw(raw: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Helper der tager enten:
    - sti til en CSV (str/Path)
    - eller en færdig DataFrame
    og returnerer et feature-DataFrame via generate_features.
    """
    if isinstance(raw, (str, Path)):
        raw_df = pd.read_csv(raw)
    elif isinstance(raw, pd.DataFrame):
        raw_df = raw.copy()
    else:
        raise TypeError(f"Unsupported raw type for _generate_from_raw: {type(raw)}")

    # Her forventes allerede kolonnerne: timestamp, open, high, low, close, volume
    df_features = generate_features(raw_df)
    return df_features


def ensure_latest(symbol: str, timeframe: str, min_rows: int = 200) -> Path:
    """
    Sikrer at vi har seneste features-fil for (symbol, timeframe).
    Hvis den ikke findes, genererer vi den ud fra raw CSV i outputs/data.
    Returnerer stien til features-filen.
    """
    raw_path = RAW_DIR / f"{symbol}_{timeframe}_raw.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw OHLCV-fil findes ikke: {raw_path}. " f"Kør scripts.fetch_raw_ohlcv_binance først."
        )

    # Generér features direkte ud fra raw CSV-stien
    df_features = _generate_from_raw(raw_path)

    if len(df_features) < min_rows:
        raise ValueError(
            f"For få rækker i features for {symbol} {timeframe}: "
            f"{len(df_features)} < min_rows={min_rows}"
        )

    # Her antager vi, at der i resten af filen er defineret et katalog til features
    features_dir = RAW_DIR / "features_auto"
    features_dir.mkdir(parents=True, exist_ok=True)

    # GEM SOM CSV (ikke parquet), så engine.read_features_auto kan læse den
    features_path = features_dir / f"{symbol}_{timeframe}_features.csv"

    df_features.to_csv(features_path, index=False)

    return features_path
