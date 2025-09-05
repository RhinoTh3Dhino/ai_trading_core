# scripts/patch_features_for_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


# De kolonner din model forventer:
REQ_COLS: List[str] = [
    "open", "high", "low", "close", "volume",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "rsi_28", "atr_14",
    "bb_upper", "bb_lower",
    "vwap", "zscore_20", "return", "pv_ratio", "regime",
    "rsi_28_z", "regime_z", "macd_z", "ema_200_z", "rsi_14_z",
    "ema_9_z", "vwap_z", "zscore_20_z", "bb_upper_z", "bb_lower_z",
    "ema_50_z", "atr_14_z", "ema_21_z",
]

# Små ‘gulvtærskler’ så vi undgår 0-division downstream
EPS_PRICE = 1e-6
EPS_ATR = 1e-6
EPS_VOL = 1.0


# ----------------------------- helpers -----------------------------
def rsi(s: pd.Series, n: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    delta = s.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=s.index).ewm(alpha=1 / n, adjust=False).mean()
    roll_down = pd.Series(down, index=s.index).ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def rolling_zscore(s: pd.Series, n: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mean = s.rolling(n, min_periods=max(2, n // 4)).mean()
    std = s.rolling(n, min_periods=max(2, n // 4)).std(ddof=0)
    out = (s - mean) / std.replace(0, np.nan)
    return out.fillna(0.0)


def read_meta_csv(path: Path) -> Tuple[Optional[dict], pd.DataFrame]:
    """
    Læser CSV hvor første linje *kan* være JSON-meta.
    Ignorerer linjer der starter med '#' (så 'Feature version ...' ikke bliver til kolonner).
    """
    first_line = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
    except Exception:
        pass

    meta = None
    skiprows = 0
    if first_line.strip().startswith("{"):
        try:
            meta = json.loads(first_line)
            skiprows = 1
        except Exception:
            meta = None
            skiprows = 0

    df = pd.read_csv(
        path,
        skiprows=skiprows,
        comment="#",              # <- vigtig: ignorer kommentartekstlinjer
        encoding="utf-8",
    )
    # Fjern spøgelseskolonner (fx "Unnamed: 35")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Fjern kolonnenavne som starter med "# "
    df = df[[c for c in df.columns if not str(c).strip().startswith("# ")]]
    return meta, df


def write_meta_csv(path: Path, meta: Optional[dict], df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(meta, dict):
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        df.to_csv(f, index=False)


def ensure_timestamp(df: pd.DataFrame, freq: str, end_utc: Optional[str]) -> pd.DataFrame:
    """
    Sørger for at 'timestamp' findes og er datetime64[ns].
    Hvis den mangler, konstrueres en syntetisk tidsakse med given frekvens.
    """
    cand_names = ["timestamp", "ts", "time", "date", "datetime"]
    have = None
    for name in cand_names:
        if name in df.columns:
            have = name
            break

    if have is not None and have != "timestamp":
        df.rename(columns={have: "timestamp"}, inplace=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        # Hvis alt blev NaT (helt ubrugeligt), generér nyt
        if df["timestamp"].isna().all():
            have = None

    if "timestamp" not in df.columns or have is None:
        # generér syntetisk tidsakse
        if end_utc:
            end = pd.Timestamp(end_utc).tz_localize("UTC").tz_convert(None)
        else:
            end = pd.Timestamp.utcnow().floor(freq).tz_localize(None)
        ts = pd.date_range(end=end, periods=len(df), freq=freq)
        df.insert(0, "timestamp", ts)

    return df


def ensure_positive_series(s: pd.Series, eps: float) -> pd.Series:
    """Sørger for ingen 0/negative værdier: NaN → bfill/ffill → eps."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.mask(~np.isfinite(s), np.nan)  # drop inf/-inf
    s = s.mask(s <= 0, np.nan).bfill().ffill().fillna(eps)
    return s.clip(lower=eps)


def compute_robust_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR(14) robust mod 0 og dårlige værdier."""
    h = ensure_positive_series(high, EPS_PRICE)
    l = ensure_positive_series(low, EPS_PRICE)
    c = ensure_positive_series(close, EPS_PRICE)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.rolling(n, min_periods=1).mean()
    atr = atr.replace([0, np.inf, -np.inf], np.nan).bfill().ffill().fillna(EPS_ATR).clip(lower=EPS_ATR)
    return atr


# ----------------------------- main patching -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Patch features to match model expectations.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV med features")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (overskrives)")
    ap.add_argument("--freq", default="H", help="Frekvens til syntetisk timestamp hvis mangler (default: H)")
    ap.add_argument("--end-utc", default=None, help="Sluttid i UTC til syntetisk timestamp (fx 2025-09-01 12:00)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)

    meta, df = read_meta_csv(inp)

    # 0) Sørg for at timestamp eksisterer
    df = ensure_timestamp(df, freq=args.freq, end_utc=args.end_utc)

    # 1) Rens primære numeriske kolonner
    for col, eps in [
        ("open", EPS_PRICE), ("high", EPS_PRICE), ("low", EPS_PRICE), ("close", EPS_PRICE),
        ("ema_9", EPS_PRICE), ("ema_21", EPS_PRICE), ("ema_50", EPS_PRICE), ("ema_200", EPS_PRICE),
        ("macd", 0.0), ("macd_signal", 0.0), ("volume", EPS_VOL),
    ]:
        if col in df.columns:
            if col == "volume":
                # volume må gerne være 0, men ikke NaN/Inf → brug EPS_VOL som fallback
                s = pd.to_numeric(df[col], errors="coerce")
                s = s.mask(~np.isfinite(s), np.nan).bfill().ffill().fillna(EPS_VOL)
                df[col] = s
            else:
                df[col] = ensure_positive_series(df[col], eps)

    # 2) Afledte kolonner
    # macd_hist
    if "macd" in df.columns and "macd_signal" in df.columns and "macd_hist" not in df.columns:
        df["macd_hist"] = pd.to_numeric(df["macd"], errors="coerce") - pd.to_numeric(df["macd_signal"], errors="coerce")

    # rsi_14
    if "rsi_14" not in df.columns and "close" in df.columns:
        df["rsi_14"] = rsi(df["close"], 14)

    # rsi_28
    if "rsi_28" not in df.columns and "close" in df.columns:
        df["rsi_28"] = rsi(df["close"], 28)

    # Bollinger bands (20, 2)
    if "close" in df.columns and (("bb_upper" not in df.columns) or ("bb_lower" not in df.columns)):
        c = pd.to_numeric(df["close"], errors="coerce")
        m = c.rolling(20, min_periods=10).mean()
        sd = c.rolling(20, min_periods=10).std(ddof=0)
        df["bb_upper"] = (m + 2 * sd).bfill().fillna(c)
        df["bb_lower"] = (m - 2 * sd).bfill().fillna(c)

    # VWAP
    h_l_c_v = {"high", "low", "close", "volume"}
    if "vwap" not in df.columns and h_l_c_v.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        tp = (high + low + close) / 3.0
        cum_pv = (tp * vol).cumsum()
        cum_v = vol.cumsum().replace(0, np.nan)
        df["vwap"] = (cum_pv / cum_v).bfill().fillna(close)

    # zscore_20 (over close)
    if "zscore_20" not in df.columns and "close" in df.columns:
        df["zscore_20"] = rolling_zscore(pd.to_numeric(df["close"], errors="coerce"), 20)

    # return (pct change på close)
    if "return" not in df.columns and "close" in df.columns:
        c = pd.to_numeric(df["close"], errors="coerce")
        df["return"] = c.pct_change().fillna(0.0)

    # pv_ratio
    if "pv_ratio" not in df.columns and "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        vma = vol.rolling(20, min_periods=5).mean()
        df["pv_ratio"] = (vol / vma.replace(0, np.nan)).fillna(1.0)

    # regime (hvis helt mangler, sæt til 0)
    if "regime" not in df.columns:
        df["regime"] = 0

    # ATR(14) – (re)beregn robust uanset om kolonnen findes, så vi er sikre på > 0
    if {"high", "low", "close"}.issubset(df.columns):
        df["atr_14"] = compute_robust_atr(df["high"], df["low"], df["close"], n=14)
    else:
        # Hvis vi ikke kan beregne: sørg for at kolonnen findes og ikke er 0
        if "atr_14" not in df.columns:
            df["atr_14"] = EPS_ATR
        else:
            s = pd.to_numeric(df["atr_14"], errors="coerce")
            df["atr_14"] = s.replace([0, np.inf, -np.inf], np.nan).bfill().ffill().fillna(EPS_ATR).clip(lower=EPS_ATR)

    # _z kolonner
    z_targets = [
        "rsi_28", "regime", "macd", "ema_200", "rsi_14", "ema_9", "vwap",
        "zscore_20", "bb_upper", "bb_lower", "ema_50", "atr_14", "ema_21",
    ]
    for base in z_targets:
        zname = f"{base}_z"
        if zname in df.columns:
            continue
        if base in df.columns:
            col = df[base]
            if col.dtype == "O":  # fx regime som tekst
                col = pd.Series(pd.factorize(col, sort=True)[0], index=col.index, dtype=float)
            df[zname] = zscore(col)
        else:
            df[zname] = 0.0

    # 3) Sørg for at alle REQ_COLS eksisterer (fyld 0.0 hvor nødvendigt)
    for c in REQ_COLS:
        if c not in df.columns and c != "timestamp":
            df[c] = 0.0

    # 4) Reorder med timestamp først
    order = ["timestamp"] + [c for c in REQ_COLS if c != "timestamp"]
    existing_pref = [c for c in order if c in df.columns]
    rest = [c for c in df.columns if c not in existing_pref]
    df = df[existing_pref + rest]

    # 5) Skriv fil
    write_meta_csv(outp, meta, df)

    # Lille sanity-opsummering (hjælper hvis noget stadig er 0)
    close = pd.to_numeric(df.get("close", pd.Series([], dtype=float)), errors="coerce")
    atr = pd.to_numeric(df.get("atr_14", pd.Series([], dtype=float)), errors="coerce")
    summary = {
        "rows": int(len(df)),
        "min_close": float(close.min()) if len(close) else None,
        "zero_close": int((close == 0).sum()) if len(close) else None,
        "min_atr_14": float(atr.min()) if len(atr) else None,
        "zero_atr_14": int((atr == 0).sum()) if len(atr) else None,
    }
    print(f"[OK] Patchet features → {outp}")
    print(f"[SUMMARY] {summary}")

    # Eventuelle manglende kolonner
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Stadig mangler: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
