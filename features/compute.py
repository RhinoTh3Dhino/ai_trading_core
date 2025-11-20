# features/compute.py
from __future__ import annotations

import numpy as np
import pandas as pd

REQ_COLS = [
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


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    d = s.diff()
    up = (d.clip(lower=0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def _z(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _rolling_z(s, n):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    m = s.rolling(n, min_periods=max(2, n // 4)).mean()
    sd = s.rolling(n, min_periods=max(2, n // 4)).std(ddof=0).replace(0, np.nan)
    return ((s - m) / sd).fillna(0.0)


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    # EMA
    for n in (9, 21, 50, 200):
        df[f"ema_{n}"] = _ema(pd.to_numeric(df["close"], errors="coerce"), n)
    # MACD(12,26,9)
    fast = _ema(pd.to_numeric(df["close"], errors="coerce"), 12)
    slow = _ema(pd.to_numeric(df["close"], errors="coerce"), 26)
    macd = fast - slow
    sig = _ema(macd, 9)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = macd - sig
    # RSI
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_28"] = _rsi(df["close"], 28)
    # ATR(14)
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.rolling(14, min_periods=1).mean()
    df["atr_14"] = (
        atr.replace([0, np.inf, -np.inf], np.nan).bfill().ffill().fillna(1e-6).clip(lower=1e-6)
    )
    # Bollinger 20,2
    m = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std(ddof=0)
    df["bb_upper"] = (m + 2 * sd).bfill().fillna(c)
    df["bb_lower"] = (m - 2 * sd).bfill().fillna(c)
    # VWAP
    tp = (h + l + c) / 3.0
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["vwap"] = ((tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)).bfill().fillna(c)
    # zscore_20 + return
    df["zscore_20"] = _rolling_z(c, 20)
    df["return"] = c.pct_change().fillna(0.0)
    # pv_ratio
    vma = vol.rolling(20, min_periods=5).mean()
    df["pv_ratio"] = (vol / vma.replace(0, np.nan)).fillna(1.0)
    # regime dummy
    if "regime" not in df:
        df["regime"] = 0
    # _z cols
    for base in [
        "rsi_28",
        "regime",
        "macd",
        "ema_200",
        "rsi_14",
        "ema_9",
        "vwap",
        "zscore_20",
        "bb_upper",
        "bb_lower",
        "ema_50",
        "atr_14",
        "ema_21",
    ]:
        zname = f"{base}_z"
        if zname not in df:
            col = df[base] if base in df else pd.Series(0.0, index=df.index)
            if col.dtype == "O":
                col = pd.Series(pd.factorize(col)[0], index=col.index, dtype=float)
            df[zname] = _z(col)
    # sørg for alle REQ_COLS
    for ccol in REQ_COLS:
        if ccol not in df:
            df[ccol] = 0.0
    return df
