# tests/test_indicators_pipeline.py
import os
import math
import numpy as np
import pytest

from data.schemas import Bar
from features.streaming_pipeline import StreamingFeaturePipeline

# Brug en gyldig venue; kan ændres via env TEST_VENUE
TEST_VENUE = os.getenv("TEST_VENUE", "binance")

def ema_series(vals, period):
    alpha = 2.0 / (period + 1.0)
    out = []
    ema = None
    for i, v in enumerate(vals):
        v = float(v)
        if ema is None:
            # seed med SMA over første 'period' hvis muligt, else første værdi
            if i + 1 >= period:
                ema = sum(vals[:period]) / period
                ema = alpha * v + (1 - alpha) * ema
            else:
                ema = v
        else:
            ema = alpha * v + (1 - alpha) * ema
        out.append(ema)
    return np.array(out, dtype=float)

def rsi_wilder(close, period=14):
    close = np.array(close, dtype=float)
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.zeros_like(close)
    avg_l = np.zeros_like(close)
    # init
    avg_g[period] = gains[1:period+1].mean()
    avg_l[period] = losses[1:period+1].mean()
    for i in range(period+1, len(close)):
        avg_g[i] = (avg_g[i-1]*(period-1) + gains[i]) / period
        avg_l[i] = (avg_l[i-1]*(period-1) + losses[i]) / period
    rs = np.divide(avg_g, avg_l, out=np.zeros_like(avg_g), where=avg_l!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def atr_wilder(high, low, close, period=14):
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = np.zeros_like(tr)
    atr[period] = tr[1:period+1].mean()
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    return atr

def make_bars(n=80, symbol="TESTUSDT", start_ts=1_700_000_000_000):
    # konstruer en mildt trendende serie
    bars = []
    ts = start_ts
    px = 100.0
    rng = np.random.default_rng(42)
    for i in range(n):
        drift = 0.1 + 0.2*math.sin(i/10.0)
        noise = rng.normal(0, 0.3)
        close = px + drift + noise
        high = max(px, close) + rng.uniform(0.0, 0.4)
        low  = min(px, close) - rng.uniform(0.0, 0.4)
        vol  = float(10 + abs(noise)*5)
        bars.append(Bar(
            venue=TEST_VENUE, symbol=symbol, interval="1m", ts=ts,
            open=float(px), high=float(high), low=float(low), close=float(close),
            volume=vol, is_final=True
        ))
        ts += 60_000
        px = close
    return bars

@pytest.mark.parametrize("period", [14, 50])
def test_ema_matches_reference(period):
    pipe = StreamingFeaturePipeline()
    bars = make_bars(120)
    closes = [b.close for b in bars]
    ref = ema_series(closes, period)

    last_val = None
    for b in bars:
        feats = pipe.update(b) or {}
        last_val = feats.get(f"ema_{period}")
    # sammenlign på de sidste ~20 punkter efter warmup
    assert last_val is not None
    assert np.isfinite(ref[-1])
    assert abs(last_val - ref[-1]) / max(1.0, abs(ref[-1])) < 0.01  # 1% tolerance

def test_rsi_and_atr_match_reference():
    pipe = StreamingFeaturePipeline()
    bars = make_bars(120)
    closes = [b.close for b in bars]
    highs  = [b.high  for b in bars]
    lows   = [b.low   for b in bars]

    rsi_ref = rsi_wilder(closes, 14)
    atr_ref = atr_wilder(highs, lows, closes, 14)

    last_rsi = last_atr = None
    for b in bars:
        feats = pipe.update(b) or {}
        last_rsi = feats.get("rsi_14")
        last_atr = feats.get("atr_14")

    assert last_rsi is not None and last_atr is not None
    assert abs(last_rsi - rsi_ref[-1]) < 1.0      # 1 RSI-point tolerance
    # ATR skala afhænger af priserne; brug relativ tolerance
    assert abs(last_atr - atr_ref[-1]) / max(1.0, abs(atr_ref[-1])) < 0.02
