# tools/generate_features_csv.py
# -*- coding: utf-8 -*-
"""
Generér features-CSV til engine.py (Fase 4) uden eksterne TA-deps.

Input:  rå candles CSV med kolonner: timestamp,open,high,low,close,volume
Output: outputs/feature_data/{symbol}_{timeframe}_features_v{version}_{YYYYMMDD}.csv
        + outputs/feature_data/{symbol}_{timeframe}_latest.csv  (pegepind-kopi)
        + outputs/feature_data/{symbol_lower}_{timeframe}_latest.csv  (ekstra pegepind)

Kør:
  python tools/generate_features_csv.py --input data/candles_btcusdt_1h.csv --symbol BTCUSDT --timeframe 1h --version 1.2
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "outputs" / "feature_data"
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Tekniske indikatorer
# -----------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Wilder’s RSI implementeret uden TA-Lib
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_ = 100.0 - (100.0 / (1.0 + rs))
    return rsi_.fillna(50.0)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    s = pd.to_numeric(series, errors="coerce")
    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    prev_close = pd.to_numeric(df["close"], errors="coerce").shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    s = pd.to_numeric(series, errors="coerce")
    ma = s.rolling(window=window, min_periods=1).mean()
    sd = s.rolling(window=window, min_periods=1).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, lower


def vwap(df: pd.DataFrame) -> pd.Series:
    # VWAP (cumulativ) = sum(typical price * vol)/sum(vol)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    typical = (high + low + close) / 3.0
    cum_pv = (typical * vol).cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)

    # Pandas FutureWarning fix: brug .bfill().ffill() i stedet for fillna(method=...)
    out = (cum_pv / cum_v).bfill().ffill()
    return out


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.rolling(window=window, min_periods=1).mean()
    std = s.rolling(window=window, min_periods=1).std(ddof=0).replace(0, np.nan)
    return ((s - m) / std).fillna(0.0)


def derive_regime(df: pd.DataFrame) -> pd.Series:
    # Simple regimes: bull (1) hvis close > EMA200, bear (-1) hvis <, ellers 0
    close = pd.to_numeric(df["close"], errors="coerce")
    ema200 = pd.to_numeric(df["ema_200"], errors="coerce")
    cond = close - ema200
    regime = np.where(cond > 0, 1, np.where(cond < 0, -1, 0))
    return pd.Series(regime, index=df.index)


# -----------------------
# Feature-produktion
# -----------------------
def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # -- timestamp til datetime (robust mod unix sekunder og ISO)
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Konverter OHLCV til numerisk
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basale EMA’er
    df["ema_9"] = ema(df["close"], 9)
    df["ema_21"] = ema(df["close"], 21)
    df["ema_50"] = ema(df["close"], 50)
    df["ema_200"] = ema(df["close"], 200)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"], 12, 26, 9)

    # RSI’er
    df["rsi_14"] = rsi(df["close"], 14)
    df["rsi_28"] = rsi(df["close"], 28)

    # ATR
    df["atr_14"] = atr(df, 14)

    # Bollinger (20, 2)
    df["bb_upper"], df["bb_lower"] = bollinger(df["close"], 20, 2.0)

    # VWAP
    df["vwap"] = vwap(df)

    # Afledte: afkast, pris-volumen ratio (signaliserende feature)
    df["return"] = pd.to_numeric(df["close"], errors="coerce").pct_change().fillna(0.0)

    # pv_ratio: prisændring * volume, normaliseret af rolling std for skala
    pv_raw = pd.to_numeric(df["close"], errors="coerce").diff().fillna(0.0) * pd.to_numeric(
        df["volume"], errors="coerce"
    ).fillna(0.0)
    pv_std = pv_raw.rolling(20, min_periods=1).std(ddof=0).replace(0, np.nan)
    df["pv_ratio"] = (pv_raw / pv_std).fillna(0.0)

    # Regime
    df["regime"] = derive_regime(df)

    # Z-scores (matcher engine “*_z”)
    base = {
        "zscore_20": zscore(df["close"], 20),  # zscore af close
        "rsi_28_z": zscore(df["rsi_28"], 20),
        "regime_z": zscore(df["regime"], 20),
        "macd_z": zscore(df["macd"], 20),
        "ema_200_z": zscore(df["ema_200"], 20),
        "rsi_14_z": zscore(df["rsi_14"], 20),
        "ema_9_z": zscore(df["ema_9"], 20),
        "vwap_z": zscore(df["vwap"], 20),
        "zscore_20_z": zscore(zscore(df["close"], 20), 20),
        "bb_upper_z": zscore(df["bb_upper"], 20),
        "bb_lower_z": zscore(df["bb_lower"], 20),
        "ema_50_z": zscore(df["ema_50"], 20),
        "atr_14_z": zscore(df["atr_14"], 20),
        "ema_21_z": zscore(df["ema_21"], 20),
    }
    for k, v in base.items():
        df[k] = v

    # Ryd ekstremiteter
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)

    # Slut-ordning af kolonner (timestamp først)
    ordered = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_200",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "rsi_28",
        "atr_14",
        "bb_upper",
        "bb_lower",
        "vwap",
        "zscore_20",
        "return",
        "pv_ratio",
        "regime",
        "rsi_28_z",
        "regime_z",
        "macd_z",
        "ema_200_z",
        "rsi_14_z",
        "ema_9_z",
        "vwap_z",
        "zscore_20_z",
        "bb_upper_z",
        "bb_lower_z",
        "ema_50_z",
        "atr_14_z",
        "ema_21_z",
    ]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    return df[cols]


# -----------------------
# I/O helpers
# -----------------------
def save_with_header(df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    header = "# " + " | ".join([f"{k}={v}" for k, v in meta.items()])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        df.to_csv(f, index=False)


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Generate feature CSV for engine.py")
    ap.add_argument(
        "--input",
        required=True,
        help="Sti til rå OHLCV CSV (timestamp,open,high,low,close,volume)",
    )
    ap.add_argument("--symbol", required=True, help="Symbol (fx BTCUSDT)")
    ap.add_argument("--timeframe", required=True, help="Tidsramme (fx 1h)")
    ap.add_argument("--version", default="1.0", help="Feature version label (fx 1.2)")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default ',')")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV findes ikke: {in_path}")

    raw = pd.read_csv(in_path, sep=args.sep)
    need = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i input: {sorted(missing)}")

    df = make_features(raw)

    date_tag = datetime.utcnow().strftime("%Y%m%d")
    # Filnavne: versioneret features i lowercase; latest i både original case og lowercase
    base_name = f"{args.symbol.lower()}_{args.timeframe}_features_v{args.version}_{date_tag}.csv"
    out_path = OUTDIR / base_name

    latest_name_exact = f"{args.symbol}_{args.timeframe}_latest.csv"
    latest_name_lower = f"{args.symbol.lower()}_{args.timeframe}_latest.csv"
    latest_path_exact = OUTDIR / latest_name_exact
    latest_path_lower = OUTDIR / latest_name_lower

    meta = {
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe,
        "version": args.version,
        "rows": len(df),
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    save_with_header(df, out_path, meta)

    # “latest”-pegepinde (Windows kan ikke altid symlinke → skriv to kopier)
    for lp in (latest_path_exact, latest_path_lower):
        try:
            if lp.exists():
                lp.unlink()
        except Exception:
            pass
        df.to_csv(lp, index=False)

    print(f"✅ Skrev: {out_path}")
    print(f"✅ Opdaterede: {latest_path_exact}")
    if latest_path_lower.as_posix() != latest_path_exact.as_posix():
        print(f"✅ Opdaterede: {latest_path_lower}")
    if len(df) < 200:
        print("⚠️ Bemærk: <200 rækker – backtest/metrikker vil være svagt informative.")


if __name__ == "__main__":
    main()
