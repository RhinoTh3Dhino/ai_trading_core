# features/auto_features.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Projekt-root
try:
    from utils.project_path import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Din eksisterende pipeline
try:
    from features.features_pipeline import generate_features, load_features
except Exception:
    generate_features = None
    load_features = None

# ccxt (valgfrit)
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None


def _latest_outpath(symbol: str, timeframe: str) -> Path:
    outdir = Path(PROJECT_ROOT) / "outputs" / "feature_data"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{symbol}_{timeframe}_latest.csv"


def _file_has_min_rows(path: Path, min_rows: int) -> bool:
    try:
        return path.exists() and len(pd.read_csv(path, nrows=min_rows + 1)) >= min_rows
    except Exception:
        return False


def _find_raw_csv_on_disk(symbol: str, timeframe: str) -> Optional[Path]:
    data_dir = Path(PROJECT_ROOT) / "data"
    if not data_dir.exists():
        return None
    # Prioritér mest specifikke navne
    patterns = [
        f"*{symbol}*{timeframe}*.csv",
        f"*{symbol}*{timeframe.replace('h','H').replace('m','M')}*.csv",
        f"*{symbol}*.csv",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(data_dir.rglob(pat))
    if not candidates:
        return None
    # Senest modificeret vinder
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _generate_from_raw(raw_path: Path) -> pd.DataFrame:
    if generate_features is None:
        raise RuntimeError("features.features_pipeline.generate_features ikke tilgængelig.")
    raw_df = pd.read_csv(raw_path)
    df_features = generate_features(
        raw_df,
        feature_config=dict(
            coerce_timestamps=True,
            patterns_enabled=True,
            target_mode="direction",
            horizon=1,
            dropna=True,
            normalize=False,
            drop_all_nan_cols=True,
        ),
    )
    return df_features


def _fetch_ohlcv_binance(
    symbol: str, timeframe: str, lookback_days: int = 180, limit_per_call: int = 1000
) -> pd.DataFrame:
    """
    Henter OHLCV via ccxt.binance (hvis installeret). Returnerer DataFrame med
    ['timestamp','open','high','low','close','volume'] i UTC ms → ISO.
    """
    if ccxt is None:
        raise RuntimeError("ccxt ikke installeret – kan ikke hente OHLCV fra Binance.")
    ex = ccxt.binance()
    # map symbol: 'BTCUSDT' → 'BTC/USDT'
    sym = f"{symbol[:-4]}/{symbol[-4:]}" if symbol.endswith(("USDT", "USDC", "BUSD")) else symbol
    now_ms = ex.milliseconds()
    since_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000
    all_rows: List[List] = []
    last = None
    while True:
        rows = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=limit_per_call)
        if not rows:
            break
        if last is not None and rows and rows[0][0] == last:
            # undgå fastlåsning hvis exchange returnerer overlap
            rows = rows[1:]
        all_rows.extend(rows)
        last = rows[-1][0] if rows else last
        if len(rows) < limit_per_call:
            break
        since_ms = rows[-1][0] + 1  # næste bar
        if len(all_rows) >= 10_000:  # hård grænse
            break
    if not all_rows:
        raise RuntimeError("Ingen OHLCV-data modtaget fra Binance.")
    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def ensure_latest(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    *,
    min_rows: int = 200,
    prefer_sources: Tuple[str, ...] = ("existing", "pipeline", "disk", "binance"),
) -> Path:
    """
    Sikrer at der findes en frisk features-CSV:
      1) existing : brug eksisterende outputs/feature_data/{sym}_{tf}_latest.csv hvis nok rækker
      2) pipeline : prøv features_pipeline.load_features(symbol, timeframe)
      3) disk     : find rå CSV under data/ og generér features
      4) binance  : ccxt fetch → generér features
    Returnerer stien til latest-filen.
    """
    out_path = _latest_outpath(symbol, timeframe)

    # 1) eksisterende
    if "existing" in prefer_sources and _file_has_min_rows(out_path, min_rows):
        return out_path

    # 2) pipeline.load_features
    if "pipeline" in prefer_sources and load_features is not None:
        try:
            df = load_features(symbol, timeframe, version_prefix=None)
            if isinstance(df, pd.DataFrame) and len(df) >= min_rows:
                df.to_csv(out_path, index=False)
                return out_path
        except Exception:
            pass

    # 3) disk/raw
    if "disk" in prefer_sources:
        raw = _find_raw_csv_on_disk(symbol, timeframe)
        if raw is not None:
            df = _generate_from_raw(raw)
            if len(df) >= min_rows:
                df.to_csv(out_path, index=False)
                return out_path

    # 4) binance (ccxt)
    if "binance" in prefer_sources:
        df_raw = _fetch_ohlcv_binance(symbol, timeframe)
        df = _generate_from_raw(df_raw)
        if len(df) >= min_rows:
            df.to_csv(out_path, index=False)
            return out_path

    raise FileNotFoundError(
        f"Kunne ikke sikre latest features for {symbol} {timeframe}. "
        f"Tjek at mindst én kilde virker (existing/pipeline/disk/binance)."
    )
