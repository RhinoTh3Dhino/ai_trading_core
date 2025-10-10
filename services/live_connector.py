# services/live_connector.py
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import DefaultDict, Dict, List

import pandas as pd

# ---- ENV & LOGGING ---------------------------------------------------------
try:
    from utils.env import load_env

    load_env()
except Exception:
    # Hvis utils.env ikke eksisterer, kører vi videre med process-miljøet
    pass

import logging


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("live_connector")

# Dæmp 3.-parts støj som standard (kan overstyres med --no-quiet)
DEFAULT_QUIET = _coerce_bool(os.getenv("QUIET_LOGS", "1"), True)


def _set_quiet_loggers(quiet: bool) -> None:
    level = logging.WARNING if quiet else LOG_LEVEL
    for name in ("websockets", "websockets.client", "websockets.protocol", "asyncio"):
        logging.getLogger(name).setLevel(level)


# ---- ORCHESTRATOR & FEATURES ----------------------------------------------
from data.feed_orchestrator import BACKUP, PRIMARY, FeedOrchestrator
from features.streaming_pipeline import \
    StreamingFeaturePipeline  # EMA(14/50), RSI(14), ATR(14), VWAP

# ---- PATHS -----------------------------------------------------------------
OUTPUTS = Path(os.getenv("OUTPUTS_DIR", "outputs/live"))
LOGS = Path(os.getenv("LOGS_DIR", "logs"))
OUTPUTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# ---- SETTINGS --------------------------------------------------------------
FLUSH_EVERY = int(
    os.getenv("LIVE_FLUSH_EVERY", "5")
)  # hvor ofte vi flusher bars til parquet
STATUS_MIN_SECS = int(
    os.getenv("LIVE_STATUS_MIN_SECS", "30")
)  # min. sekunder mellem statuslinjer (samlet)
WRITE_METRICS = _coerce_bool(
    os.getenv("LIVE_WRITE_METRICS", "1"), True
)  # skriv feed_metrics.jsonl
LAG_WINDOW = int(os.getenv("LAG_WINDOW", "20"))  # glidende vindue for bar_close_lag_ms
SCHEMA_VERSION = os.getenv("LIVE_SCHEMA_VERSION", "stream-mvp-1")  # Parquet metadata

FEATURE_COLS = ["ema_14", "ema_50", "rsi_14", "vwap", "atr_14"]
BASE_COLS = ["ts", "open", "high", "low", "close", "volume"]
ALL_COLS = BASE_COLS + FEATURE_COLS


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_parquet_with_meta(df: pd.DataFrame, path: Path, schema_version: str):
    """
    Skriv Parquet med metadata hvis pyarrow er tilgængelig.
    Fallback: almindelig pandas + sidecar .meta.json
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[b"schema_version"] = schema_version.encode("utf-8")
        md[b"columns"] = ",".join(df.columns).encode("utf-8")
        table = table.replace_schema_metadata(md)
        pq.write_table(table, path)
    except Exception:
        # Fallback
        df.to_parquet(path, index=False)
        try:
            with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"schema_version": schema_version, "columns": list(df.columns)}, f
                )
        except Exception:
            log.debug("Kunne ikke skrive sidecar metadata for %s", path)


async def main(symbols: List[str], interval: str = "1m", quiet: bool = DEFAULT_QUIET):
    _set_quiet_loggers(quiet)

    # Startup-banner
    log.info("Starter live_connector")
    log.info(
        "ENV=%s  SYMBOLS=%s  INTERVAL=%s", os.getenv("ENV", "DEV"), symbols, interval
    )
    log.info("PRIMARY venues=%s  BACKUP venues=%s", PRIMARY, BACKUP)
    log.info(
        "OUTPUTS=%s  FLUSH_EVERY=%s  STATUS_MIN_SECS=%s  LOG_LEVEL=%s  QUIET=%s",
        str(OUTPUTS),
        FLUSH_EVERY,
        STATUS_MIN_SECS,
        logging.getLevelName(LOG_LEVEL),
        quiet,
    )

    # Start orchestrator
    orch = FeedOrchestrator(symbols, interval, min_active=2)
    q = await orch.run()

    # Streaming features (kun lukkede barer)
    pipe = StreamingFeaturePipeline()

    # Parquet buffers pr. symbol
    parquet_files: Dict[str, Path] = {
        s: OUTPUTS / f"{s}_{interval}.parquet" for s in symbols
    }
    dfs: Dict[str, pd.DataFrame] = {s: pd.DataFrame(columns=ALL_COLS) for s in symbols}
    bar_counts: Dict[str, int] = {s: 0 for s in symbols}

    # Status-rate-limit (samlet status for alle symbols)
    last_status_wall: float = 0.0

    # Glidende “close lag”-vindue per symbol (kun for lukkede bars)
    close_lag_windows: DefaultDict[str, deque] = defaultdict(
        lambda: deque(maxlen=LAG_WINDOW)
    )

    log.info("Venter på første bar ...")
    try:
        while True:
            bar = await q.get()  # Blokerer indtil næste besked

            # --- Metrics for ALLE bars (også u-lukkede) ---------------------
            now_ms = _now_ms()

            # Forsøg at bruge et “event/source” timestamp hvis Bar har det; fallback til bar.ts
            event_ts = None
            for cand in ("event_ts", "source_ts", "ingest_ts", "recv_ts"):
                if hasattr(bar, cand):
                    try:
                        event_ts = int(getattr(bar, cand))
                        break
                    except Exception:
                        pass
            if event_ts is None:
                try:
                    event_ts = int(bar.ts)
                except Exception:
                    event_ts = now_ms

            try:
                transport_latency_ms = max(0, now_ms - int(event_ts))
            except Exception:
                transport_latency_ms = None

            try:
                bar_close_lag_ms = now_ms - int(bar.ts)  # bar.ts = slut-tid i ms
            except Exception:
                bar_close_lag_ms = None

            if WRITE_METRICS:
                payload = {
                    "ts": now_ms,
                    "symbol": getattr(bar, "symbol", "UNKNOWN"),
                    "venue": getattr(bar, "venue", "unknown"),
                    "bar_ts": int(getattr(bar, "ts", now_ms)),
                    "is_final": bool(getattr(bar, "is_final", False)),
                    "transport_latency_ms": transport_latency_ms,
                    "bar_close_lag_ms": bar_close_lag_ms,
                    "event_ts": int(event_ts) if event_ts is not None else None,
                }
                try:
                    with open(LOGS / "feed_metrics.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")
                except Exception:
                    pass

            # --- Kun LUKKEDE barer skrives til Parquet og tælles -------------
            if not getattr(bar, "is_final", False):
                continue

            feats = pipe.update(bar) or {}

            # Lazy init hvis et symbol dukker op, vi ikke forventede
            if bar.symbol not in dfs:
                dfs[bar.symbol] = pd.DataFrame(columns=ALL_COLS)
                parquet_files[bar.symbol] = OUTPUTS / f"{bar.symbol}_{interval}.parquet"
                bar_counts[bar.symbol] = 0

            # Row til skrivning
            row = {
                "ts": int(bar.ts),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for k in FEATURE_COLS:
                row[k] = (
                    float(feats[k]) if k in feats and feats[k] is not None else None
                )

            df = dfs[bar.symbol]
            df.loc[len(df)] = [row.get(c, None) for c in ALL_COLS]
            bar_counts[bar.symbol] += 1

            # Latenser (kun på lukkede barer for status/lag)
            if isinstance(bar_close_lag_ms, (int, float)):
                close_lag_windows[bar.symbol].append(int(bar_close_lag_ms))

            # Periodisk flush til disk (Parquet) pr. symbol
            if bar_counts[bar.symbol] % FLUSH_EVERY == 0:
                try:
                    df_sorted = df.sort_values("ts").drop_duplicates(
                        subset=["ts"], keep="last"
                    )
                    _write_parquet_with_meta(
                        df_sorted, parquet_files[bar.symbol], SCHEMA_VERSION
                    )
                except Exception as e:
                    log.warning("Parquet flush fejlede for %s: %s", bar.symbol, repr(e))

            # Samlet statuslinje med rate limit
            now = time.time()
            if (now - last_status_wall) >= STATUS_MIN_SECS:
                # p99 pr. venue
                try:
                    p99_latencies = {v: orch.p99[v].p99() for v in orch.p99}
                except Exception:
                    p99_latencies = {}
                # bars pr. symbol kompakt
                bars_summary = ", ".join(
                    f"{s}:{bar_counts.get(s, 0)}" for s in sorted(bar_counts)
                )
                # glidende gennemsnit af bar_close_lag_ms pr. symbol
                lag_pairs = []
                for s in sorted(bar_counts):
                    window = close_lag_windows.get(s, [])
                    if window:
                        avg = int(sum(window) / len(window))
                        lag_pairs.append(f"{s}:{avg}")
                    else:
                        lag_pairs.append(f"{s}:na")
                lag_str = "{" + ", ".join(lag_pairs) + "}"

                log.info(
                    "Status: bars={%s}  p99(ms)=%s  lag_ms≈%s  active=%s",
                    bars_summary,
                    p99_latencies,
                    lag_str,
                    orch.active,
                )
                last_status_wall = now

            # (valgfrit) publish (bar, feats) på en signal-bus
            # from services.signal_bus import publish
            # await publish({"bar": bar.dict(), "features": feats})

    except asyncio.CancelledError:
        log.info("Live connector annulleret (Cancel).")
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – lukker pænt ned.")
    except Exception as e:
        log.exception("Uventet fejl i live connector: %s", repr(e))
    finally:
        # Stop WS-tasks pænt (hvis orchestratoren har shutdown())
        try:
            shutdown = getattr(orch, "shutdown", None)
            if callable(shutdown):
                await shutdown()
        except Exception as e:
            log.debug("Orchestrator shutdown gav en advarsel: %s", repr(e))

        # Final flush
        for sym, df in dfs.items():
            try:
                if not df.empty:
                    df_sorted = df.sort_values("ts").drop_duplicates(
                        subset=["ts"], keep="last"
                    )
                    _write_parquet_with_meta(
                        df_sorted, parquet_files[sym], SCHEMA_VERSION
                    )
            except Exception as e:
                log.warning("Final flush fejlede for %s: %s", sym, repr(e))
        log.info("Live connector stoppet.")


if __name__ == "__main__":
    # Windows: ren Ctrl+C
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    import argparse
    import re

    parser = argparse.ArgumentParser()
    # Acceptér både: --symbols BTCUSDT ETHUSDT  og  --symbols "BTCUSDT,ETHUSDT"
    parser.add_argument(
        "--symbols", nargs="+", default=[os.getenv("SYMBOLS", "BTCUSDT")]
    )
    parser.add_argument("--interval", default=os.getenv("BAR_INTERVAL", "1m"))
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=DEFAULT_QUIET,
        help="Dæmp 3.-parts logger-støj (default fra env QUIET_LOGS=1).",
    )
    parser.add_argument(
        "--no-quiet",
        dest="quiet",
        action="store_false",
        help="Vis detaljerede websockets/asyncio logs.",
    )
    args = parser.parse_args()

    # Flad ud og split på komma, hvis nogen entries er streng med komma
    raw_parts: List[str] = []
    for item in args.symbols:
        if isinstance(item, str) and ("," in item):
            raw_parts.extend([p for p in item.split(",") if p.strip()])
        else:
            raw_parts.append(str(item))

    # Normalisér og valider
    SYM_RE = re.compile(r"^[A-Z0-9_\-./]{4,}$")  # Binance/Bybit tickers uden særtegn
    symbols: List[str] = []
    bad: List[str] = []
    for s in raw_parts:
        ss = s.strip().upper()
        if ss and SYM_RE.match(ss):
            symbols.append(ss)
        else:
            bad.append(s)

    if bad:
        log.error("Ugyldige symboler i --symbols: %s", bad)
        sys.exit(2)

    try:
        asyncio.run(main(symbols, args.interval, quiet=args.quiet))
    except KeyboardInterrupt:
        log.info("Afslutter (Ctrl+C).")
        try:
            sys.exit(0)
        except SystemExit:
            pass
