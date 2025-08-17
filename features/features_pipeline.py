# features/features_pipeline.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime

from utils.project_path import PROJECT_ROOT
from features.patterns import add_all_patterns


# ============================================================
# Helper-funktioner (små, rene og lette at teste)
# ============================================================

def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner: {missing} (fandt: {list(df.columns)})")


def _map_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns and "timestamp" not in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})
    if "Timestamp" in out.columns and "timestamp" not in out.columns:
        out = out.rename(columns={"Timestamp": "timestamp"})
    return out


def _to_timestamp(df: pd.DataFrame, coerce: bool) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        # Validering fanges i _ensure_required_columns; her undgås KeyError
        return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce" if coerce else "raise")
    if out["timestamp"].isna().any():
        raise ValueError("Ugyldige timestamp-værdier fundet.")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=close.index).rolling(length).mean()
    avg_loss = pd.Series(loss, index=close.index).rolling(length).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def _select_columns(df: pd.DataFrame, include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]) -> pd.DataFrame:
    out = df.copy()
    if exclude:
        drop_cols = [c for c in exclude if c in out.columns]
        out = out.drop(columns=drop_cols)
    if include:
        keep = [c for c in include if c in out.columns]
        # bevar target/regime hvis de allerede findes
        base_keep = set(keep)
        for extra in ("target", "regime"):
            if extra in out.columns:
                base_keep.add(extra)
        out = out.loc[:, [c for c in out.columns if c in base_keep]]
    return out


def _feature_match(df: pd.DataFrame, expected: Iterable[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in expected if c not in df.columns]
    return (len(missing) == 0, missing)


def _normalize_minmax(df: pd.DataFrame, skip_cols: Iterable[str] = ()) -> pd.DataFrame:
    """
    Robust min-max normalisering:
    - Fuld-NaN kolonner bliver 0.0 (stabilt output).
    - Konstante kolonner bliver 0.0.
    - Kolonnevis assignment → undgår FutureWarnings og bevarer NaN hvor de var.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns.difference(list(skip_cols))
    for c in num_cols:
        col = out[c]
        if col.isna().all():
            out[c] = 0.0
            continue
        cmin = col.min(skipna=True)
        cmax = col.max(skipna=True)
        if pd.isna(cmin) or pd.isna(cmax):
            out[c] = 0.0
            continue
        denom = cmax - cmin
        if denom == 0:
            out[c] = 0.0
        else:
            out[c] = (col - cmin) / denom
    return out


# ============================================================
# Offentlig API
# ============================================================

def generate_features(df: pd.DataFrame, feature_config: dict | None = None) -> pd.DataFrame:
    """
    Samlet pipeline til at beregne tekniske indikatorer + pattern-features.

    Parameters
    ----------
    df : pd.DataFrame
        Skal som minimum indeholde OHLCV + timestamp (evt. 'datetime'/'Timestamp' mappes).
    feature_config : dict | None
        Konfiguration (alle nøgler er valgfrie):
        - require_columns: list[str] (default: ["timestamp", "close", "volume", "open", "high", "low"])
        - coerce_timestamps: bool (default: True)
        - patterns_enabled: bool (default: True)
        - expected_features: list[str] (default: None) -> validerer, rejser ValueError ved mismatch
        - include: list[str] (default: None)
        - exclude: list[str] (default: None)
        - dropna: bool (default: True)
        - target_mode: "direction" | "regression" | "none" (default: "direction")
        - horizon: int (default: 1)  # bruges til target
        - normalize: bool (default: False)  # min-max på numeriske kolonner
        - drop_all_nan_cols: bool (default: True)  # fjern featurekolonner der er 100% NaN før dropna

    Returns
    -------
    pd.DataFrame
        Input med ekstra feature-kolonner og evt. target/regime.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame er tom – kan ikke generere features.")

    cfg = feature_config.copy() if isinstance(feature_config, dict) else {}
    require_cols: List[str] = cfg.get("require_columns", ["timestamp", "close", "volume", "open", "high", "low"])
    coerce_ts: bool = bool(cfg.get("coerce_timestamps", True))
    patterns_enabled: bool = bool(cfg.get("patterns_enabled", True))
    expected_features: Optional[List[str]] = cfg.get("expected_features")
    include: Optional[List[str]] = cfg.get("include")
    exclude: Optional[List[str]] = cfg.get("exclude")
    dropna: bool = bool(cfg.get("dropna", True))
    target_mode: str = cfg.get("target_mode", "direction")
    horizon: int = int(cfg.get("horizon", 1))
    normalize: bool = bool(cfg.get("normalize", False))
    drop_all_nan_cols: bool = bool(cfg.get("drop_all_nan_cols", True))

    out = df.copy()

    # --- Kolonne-mapping ---
    out = _map_timestamp_columns(out)

    # --- Tjek basis-kolonner ---
    _ensure_required_columns(out, require_cols)

    # --- Konverter til datetime & sortér ---
    out = _to_timestamp(out, coerce=coerce_ts)

    # --- Konverter til numeric ---
    out = _to_numeric(out, cols=["open", "high", "low", "close", "volume"])

    # === TEKNISKE INDIKATORER ===
    out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()

    for rsi_n in (14, 28):
        out[f"rsi_{rsi_n}"] = _rsi(out["close"], rsi_n)

    out["macd"], out["macd_signal"] = _macd(out["close"])
    # VWAP (kumulativ, robust ift. 0-division)
    out["vwap"] = (out["close"] * out["volume"]).cumsum() / (out["volume"].cumsum() + 1e-9)

    out["atr_14"] = _atr(out["high"], out["low"], out["close"], window=14)
    out["return"] = out["close"].pct_change().fillna(0.0)
    out["pv_ratio"] = out["close"] / (out["volume"] + 1e-9)
    out["regime"] = (out["ema_9"] > out["ema_21"]).astype(int)

    mid, up, low = _bollinger(out["close"], window=20, num_std=2.0)
    out["bb_middle"] = mid
    out["bb_upper"] = up
    out["bb_lower"] = low

    # === PATTERN FEATURES (valgfrie) ===
    if patterns_enabled:
        try:
            out = add_all_patterns(out, breakout_lookback=20, vol_mult=2.0)
        except Exception as e:
            print(f"[WARN] add_all_patterns fejlede: {e}")

    # Konsistent kolonnenavn
    if "vol_spike" in out.columns and "volume_spike" not in out.columns:
        out = out.rename(columns={"vol_spike": "volume_spike"})

    # === Target (3 modes) ===
    if target_mode not in ("direction", "regression", "none"):
        raise ValueError(f"Ugyldig target_mode: {target_mode}")

    if not any([col for col in out.columns if str(col).startswith("target")]):
        if target_mode == "direction":
            fwd = out["close"].shift(-horizon)
            out["target"] = (fwd > out["close"]).astype(int)
        elif target_mode == "regression":
            fwd = out["close"].shift(-horizon)
            out["target"] = ((fwd - out["close"]) / out["close"]).fillna(0.0)
        else:
            pass  # "none": bevidst ingen target

    # === Fjern fuldt-NaN featurekolonner før normalisering/dropna (Option B) ===
    if drop_all_nan_cols:
        numeric_cols = out.select_dtypes(include=["number"]).columns.difference(["target", "regime"])
        all_nan_cols = [c for c in numeric_cols if out[c].isna().all()]
        if all_nan_cols:
            out = out.drop(columns=all_nan_cols)

    # === Valgfri normalisering (efter target/labels er sat) ===
    if normalize:
        out = _normalize_minmax(out, skip_cols=("target", "regime"))

    # === Valgfri select/filter ===
    out = _select_columns(out, include=include, exclude=exclude)

    # === Valgfri feature-match validering ===
    if expected_features:
        ok, missing = _feature_match(out, expected_features)
        if not ok:
            raise ValueError(f"Feature-match fejlede. Mangler: {missing}")

    # === Ryd op ===
    if dropna:
        out = out.dropna().reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out


def save_features(df: pd.DataFrame, symbol: str, timeframe: str, version: str = "v1") -> str:
    if df is None or df.empty:
        raise ValueError("Kan ikke gemme tom DataFrame som features.")

    today = datetime.now().strftime("%Y%m%d")
    filename = f"{symbol.lower()}_{timeframe}_features_{version}_{today}.csv"
    output_dir = Path(PROJECT_ROOT) / "outputs" / "feature_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / filename
    df.to_csv(full_path, index=False)
    print(f"✅ Features gemt: {full_path}")
    return str(full_path)


def load_features(symbol: str, timeframe: str, version_prefix: str = "v1") -> pd.DataFrame:
    folder = Path(PROJECT_ROOT) / "outputs" / "feature_data"
    if not folder.exists():
        raise FileNotFoundError(f"Feature-mappe mangler: {folder}")

    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.name.startswith(f"{symbol.lower()}_{timeframe}_features_{version_prefix}")
    ]
    if not files:
        raise FileNotFoundError(f"Ingen feature-filer fundet for {symbol} {timeframe} ({version_prefix})")

    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    newest_file = files[0]
    print(f"📥 Indlæser features: {newest_file}")
    return pd.read_csv(newest_file)


if __name__ == "__main__":
    # Lille CLI-agtig røntgen for hurtig manuel test og coverage-branching
    path = Path(PROJECT_ROOT) / "data" / "BTCUSDT_1h_with_target.csv"

    # Test-case 1: Success eller demo-DF fallback
    if path.exists():
        df = pd.read_csv(path)
    else:
        print(f"[INFO] Testfil mangler: {path} – bruger demo-DF.")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "open": np.linspace(100, 110, 50),
            "high": np.linspace(101, 112, 50),
            "low": np.linspace(99, 108, 50),
            "close": np.linspace(100, 111, 50) + np.random.randn(50) * 0.3,
            "volume": np.random.randint(10, 50, size=50),
        })

    try:
        features = generate_features(df, feature_config={
            "coerce_timestamps": True,
            "patterns_enabled": True,
            "target_mode": "direction",
            "horizon": 1,
            "dropna": True,
            "normalize": True,
            "drop_all_nan_cols": True,
        })
        print("✅ Features genereret:", features.shape)
    except Exception as e:
        print("❌ Fejl ved feature-generering:", e)

    # Test-case 2: Manglende kolonner
    try:
        df_bad = pd.DataFrame({"timestamp": [], "close": []})
        generate_features(df_bad)
    except Exception as e:
        print("[TEST] Forventet fejl:", e)
