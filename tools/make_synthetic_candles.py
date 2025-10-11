## tools/make_synthetic_candles.py


# -*- coding: utf-8 -*-
"""
Syntetiske OHLCV-candles til udvikling, tests og Fase-4 persistens.
- Understøtter regimer: bull / bear / sideways
- Fleksibel tidsfrekvens (1m/5m/15m/30m/1h/4h/1d)
- Deterministisk seed for reproducerbarhed
- Robust pris- og wick-generering (high/low konsistente)
- Volumendynamik pr. regime

Output: CSV med kolonner: timestamp,open,high,low,close,volume

Eksempler (PowerShell):
  # 2000 bars, tydelige regimer i 1H
  python tools/make_synthetic_candles.py `
    --out data\candles_btcusdt_1h.csv `
    --rows 2000 `
    --freq 1h `
    --regimes bull:700 sideways:300 bear:700 sideways:300 `
    --start "2024-01-01 00:00:00" `
    --start-price 40000

  # 7 dage i 5m med mere rolig drift og færre shocks
  python tools/make_synthetic_candles.py `
    --out data\candles_btcusdt_5m.csv `
    --rows 2016 `
    --freq 5m `
    --mu-bull 0.0006 --mu-bear -0.0006 --sigma 0.0015 `
    --shock-prob 0.002 --shock-mult 3.0
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

# ---------- Konstanter & helpers ----------
_FREQ_TO_MIN = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


@dataclass
class RegimeParams:
    name: str
    mu: float  # gennemsnitlig log-afkast pr. bar (approks.)
    sigma: float  # standardafvigelse pr. bar
    vol_lo: float  # volumeninterval (lav)
    vol_hi: float  # volumeninterval (høj)
    wick_amp: float  # amplitude for wick-udslag


def _parse_regime_spec(spec: List[str]) -> List[Tuple[str, int]]:
    """
    Parse --regimes argumenter af formen ["bull:600", "sideways:300", "bear:700"].
    """
    out: List[Tuple[str, int]] = []
    for tok in spec:
        if ":" not in tok:
            raise ValueError(f"Ugyldig regime-token '{tok}'. Forvent '<navn>:<bars>'")
        name, num = tok.split(":", 1)
        name = name.strip().lower()
        if name not in {"bull", "bear", "sideways"}:
            raise ValueError(f"Ukendt regime '{name}'. Tilladt: bull, bear, sideways")
        n = int(num)
        if n <= 0:
            raise ValueError(f"Regime-længde skal være > 0 (fik {n})")
        out.append((name, n))
    return out


def _expand_schedule(total: int, blocks: List[Tuple[str, int]]) -> List[str]:
    seq: List[str] = []
    for nm, ln in blocks:
        seq.extend([nm] * ln)
    if len(seq) < total:
        seq.extend([blocks[-1][0]] * (total - len(seq)))
    return seq[:total]


def _minutes_for_freq(freq: str) -> int:
    f = freq.lower()
    if f not in _FREQ_TO_MIN:
        raise ValueError(f"Ukendt frekvens '{freq}'. Vælg én af: {', '.join(_FREQ_TO_MIN)}")
    return _FREQ_TO_MIN[f]


def _next_time(ts: datetime, freq: str) -> datetime:
    mins = _minutes_for_freq(freq)
    return ts + timedelta(minutes=mins)


def _gauss(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)


def _apply_shock(ret: float, prob: float, mult: float) -> float:
    if prob <= 0 or mult <= 1:
        return ret
    if random.random() < prob:
        direction = 1.0 if ret >= 0 else -1.0
        return ret * (mult * direction)
    return ret


def _bounded(value: float, min_v: float = 1e-9) -> float:
    return max(min_v, value)


def _round2(x: float) -> float:
    return round(x, 2)


def _round3(x: float) -> float:
    return round(x, 3)


# ---------- Generator ----------
def gen_candles(
    rows: int,
    start_iso: str,
    freq: str,
    start_price: float,
    seed: int,
    regimes: List[Tuple[str, int]],
    # basis-drift/vol for hver regime (kan overskrives pr. regime nedenfor)
    mu_bull: float,
    mu_bear: float,
    mu_sideways: float,
    sigma: float,
    wick_amp: float,
    # shocks
    shock_prob: float,
    shock_mult: float,
    # volumen base
    vol_base_lo: float,
    vol_base_hi: float,
    vol_bull_mult: float,
    vol_bear_mult: float,
    vol_side_mult: float,
) -> Iterable[Tuple[str, float, float, float, float, float]]:
    """
    Generator for syntetiske candles.

    Returnerer tuples: (timestamp, open, high, low, close, volume)
    """
    random.seed(seed)

    # Regime-parametre (kan finjusteres her)
    regime_cfg = {
        "bull": RegimeParams(
            "bull",
            mu_bull,
            sigma,
            vol_base_lo * vol_bull_mult,
            vol_base_hi * vol_bull_mult,
            wick_amp,
        ),
        "bear": RegimeParams(
            "bear",
            mu_bear,
            sigma,
            vol_base_lo * vol_bear_mult,
            vol_base_hi * vol_bear_mult,
            wick_amp,
        ),
        "sideways": RegimeParams(
            "sideways",
            mu_sideways,
            sigma,
            vol_base_lo * vol_side_mult,
            vol_base_hi * vol_side_mult,
            wick_amp,
        ),
    }

    sched = _expand_schedule(rows, regimes)
    t0 = datetime.fromisoformat(start_iso)
    t = t0
    price = float(start_price)

    for i in range(rows):
        reg = regime_cfg[sched[i]]

        # Brug log-normal-walk via additive approx. på logrets (lille-sigma antagelse)
        # ret ~ N(mu, sigma). Shock anvendes sjældent, men kan skabe tydeligere bevægelse.
        ret = _gauss(reg.mu, reg.sigma)
        ret = _apply_shock(ret, shock_prob, shock_mult)

        o = price
        # transformér ret til relativ ændring på prisniveau
        c = _bounded(o * math.exp(ret))
        base_spread = abs(c - o)

        # Wick generering – proportional med spread og en lille ekstra amplitude
        wick = reg.wick_amp
        hi = max(o, c) * (
            1.0 + max(0.0, 0.25 * (base_spread / max(o, 1.0)) + random.uniform(0.0, wick))
        )
        lo = min(o, c) * (
            1.0 - max(0.0, 0.25 * (base_spread / max(o, 1.0)) + random.uniform(0.0, wick))
        )
        lo = _bounded(lo)  # undgå negative priser

        # Volumen afhænger af regime
        vol = random.uniform(reg.vol_lo, reg.vol_hi)

        # Konsistens-sikring
        H = max(hi, o, c)
        L = min(lo, o, c)

        ts_str = t.strftime("%Y-%m-%d %H:%M:%S")
        yield (ts_str, _round2(o), _round2(H), _round2(L), _round2(c), _round3(vol))

        # næste bar
        price = c
        t = _next_time(t, freq)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Syntetiske OHLCV-candles med regimer")
    ap.add_argument("--out", default="data/candles_btcusdt_1h.csv", help="Output CSV-sti")
    ap.add_argument("--rows", type=int, default=2000, help="Antal bars at generere")
    ap.add_argument("--freq", default="1h", choices=list(_FREQ_TO_MIN.keys()), help="Bar-frekvens")
    ap.add_argument("--start", default="2024-01-01 00:00:00", help="Starttid (ISO, lokal uden TZ)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--start-price", type=float, default=40000.0, help="Startpris")
    ap.add_argument(
        "--regimes",
        nargs="+",
        default=["bull:700", "sideways:300", "bear:700", "sideways:300"],
        help="Liste af '<navn>:<bars>' (navn i {bull,bear,sideways})",
    )

    # Drift/vol-parametre (kan tunes)
    ap.add_argument(
        "--mu-bull",
        type=float,
        default=0.0012,
        help="Bull gennemsnitlig log-ret pr. bar",
    )
    ap.add_argument(
        "--mu-bear",
        type=float,
        default=-0.0012,
        help="Bear gennemsnitlig log-ret pr. bar",
    )
    ap.add_argument(
        "--mu-sideways",
        type=float,
        default=0.0,
        help="Sideways gennemsnitlig log-ret pr. bar",
    )
    ap.add_argument("--sigma", type=float, default=0.0020, help="Std.dev for log-ret pr. bar")
    ap.add_argument(
        "--wick-amp",
        type=float,
        default=0.0020,
        help="Wick amplitude (typisk 0.001-0.004)",
    )

    # Shocks (sjældne udslag)
    ap.add_argument(
        "--shock-prob",
        type=float,
        default=0.003,
        help="Sandsynlighed for shock pr. bar (0-1)",
    )
    ap.add_argument(
        "--shock-mult",
        type=float,
        default=2.5,
        help="Multiplikator på ret under shock (>=1)",
    )

    # Volumen
    ap.add_argument("--vol-base-lo", type=float, default=30.0, help="Basis minimumsvolumen")
    ap.add_argument("--vol-base-hi", type=float, default=800.0, help="Basis maksimumsvolumen")
    ap.add_argument("--vol-bull-mult", type=float, default=1.1, help="Bull volumemultiplikator")
    ap.add_argument("--vol-bear-mult", type=float, default=1.2, help="Bear volumemultiplikator")
    ap.add_argument("--vol-side-mult", type=float, default=1.0, help="Sideways volumemultiplikator")

    args = ap.parse_args()

    blocks = _parse_regime_spec(args.regimes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in gen_candles(
            rows=args.rows,
            start_iso=args.start,
            freq=args.freq,
            start_price=args.start_price,
            seed=args.seed,
            regimes=blocks,
            mu_bull=args.mu_bull,
            mu_bear=args.mu_bear,
            mu_sideways=args.mu_sideways,
            sigma=args.sigma,
            wick_amp=args.wick_amp,
            shock_prob=args.shock_prob,
            shock_mult=args.shock_mult,
            vol_base_lo=args.vol_base_lo,
            vol_base_hi=args.vol_base_hi,
            vol_bull_mult=args.vol_bull_mult,
            vol_bear_mult=args.vol_bear_mult,
            vol_side_mult=args.vol_side_mult,
        ):
            w.writerow(row)

    print(f"✅ Skrev rå candles: {out_path}  (rækker: {args.rows}, freq: {args.freq})")


if __name__ == "__main__":
    main()
