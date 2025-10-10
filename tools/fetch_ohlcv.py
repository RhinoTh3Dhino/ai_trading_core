## tools/fetch_ohlcv.py


# -*- coding: utf-8 -*-
"""
Hent OHLCV fra OKX eller Kraken (spot) uden eksterne afhængigheder.
Output: CSV med kolonner: timestamp,open,high,low,close,volume

Eksempler:
  OKX (seneste 1000 1H):
    python tools/fetch_ohlcv.py --exchange okx --symbol BTC-USDT --timeframe 1h --limit 1000 --out data/candles_okx_btcusdt_1h.csv

  Kraken (seneste 1000 1H), par = XBTUSDT:
    python tools/fetch_ohlcv.py --exchange kraken --symbol XBTUSDT --timeframe 1h --limit 1000 --out data/candles_kraken_xbtusdt_1h.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests  # type: ignore
except Exception:
    print("Denne fetcher kræver 'requests'. Kør: pip install requests", file=sys.stderr)
    sys.exit(2)

OKX_BASE = "https://www.okx.com"
KRAKEN_BASE = "https://api.kraken.com"

_TF_OKX = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}
_TF_KRAKEN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def fetch_okx(symbol: str, timeframe: str, limit: int = 1000):
    bar = _TF_OKX.get(timeframe)
    if not bar:
        raise ValueError(f"OKX understøtter ikke timeframe={timeframe}")
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": symbol, "bar": bar, "limit": str(limit)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX fejl: {data}")
    rows = []
    # Data returneres nyeste først: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    for item in reversed(data.get("data", [])):
        ts = int(item[0])
        o, h, l, c = map(float, item[1:5])
        vol = (
            float(item[6])
            if len(item) > 6 and item[6] not in ("", None)
            else float(item[5])
        )
        rows.append([_to_iso(ts), o, h, l, c, vol])
    return rows


def fetch_kraken(pair: str, timeframe: str, limit: int = 1000):
    itv = _TF_KRAKEN.get(timeframe)
    if not itv:
        raise ValueError(f"Kraken understøtter ikke timeframe={timeframe}")
    url = f"{KRAKEN_BASE}/0/public/OHLC"
    params = {"pair": pair, "interval": itv}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken fejl: {data['error']}")
    # Result-key er par-navnet; hvert element: [time, open, high, low, close, vwap, volume, count]
    result_key = next(iter(data["result"].keys() - {"last"}))
    ohlc = data["result"][result_key]
    rows = []
    for item in ohlc[-limit:]:
        ts_sec = int(item[0])
        o, h, l, c = float(item[1]), float(item[2]), float(item[3]), float(item[4])
        vol = float(item[6])  # "volume"
        rows.append(
            [
                datetime.utcfromtimestamp(ts_sec).strftime("%Y-%m-%d %H:%M:%S"),
                o,
                h,
                l,
                c,
                vol,
            ]
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True, choices=["okx", "kraken"])
    # OKX: "BTC-USDT", Kraken: "XBTUSDT" (eller XBTUSD osv.)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.exchange == "okx" and "-" not in args.symbol:
        print("OKX symbol skal være som 'BTC-USDT' (med bindestreg).", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.exchange == "okx":
        rows = fetch_okx(args.symbol, args.timeframe, args.limit)
    else:
        rows = fetch_kraken(args.symbol, args.timeframe, args.limit)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow(r)
    print(f"✅ Skrev: {out_path} (rækker: {len(rows)})")


if __name__ == "__main__":
    main()
