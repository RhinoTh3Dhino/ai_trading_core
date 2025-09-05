# -*- coding: utf-8 -*-
# live.py – Paper/live “daemon” der opdaterer GUI-logfilerne pr. bar
from __future__ import annotations

import os, json, time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- Data + features + model ---
from data.live_feed import fetch_ohlcv_df
from features.compute import compute_features
from engines.inference import (
    load_torch_model, load_scaler, load_feature_order, run_inference
)

# --- Broker (Paper) ---
from bot.brokers.paper import PaperBroker  # én kanonisk broker


# =========================
# Paths + helpers
# =========================
ROOT = Path(__file__).resolve().parent           # <- projektrod (fix)
LOGS = Path(os.getenv("LOG_DIR", ROOT / "logs"))
LOGS.mkdir(parents=True, exist_ok=True)
API_DIR = ROOT / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_CSV = LOGS / "equity.csv"
FILLS_CSV = LOGS / "fills.csv"
SIGNALS_CSV = LOGS / "signals.csv"
DAILY_METRICS_CSV = LOGS / "daily_metrics.csv"
LAST_TS_FILE = LOGS / ".last_bar_ts"
SIM_SIGNALS_JSON = API_DIR / "sim_signals.json"  # GUI læser denne til "Live signaler"

def _ts_iso(dt: pd.Timestamp | datetime) -> str:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def _finite(x: float, fallback: float = 0.0) -> float:
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else float(fallback)
    except Exception:
        return float(fallback)

def _append_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        for r in rows:
            safe = [str(_finite(x) if isinstance(x, (float, np.floating)) else x) for x in r]
            f.write(",".join(map(str, safe)) + "\n")

def _write_json_list_safely(path: Path, payload: List[Dict]) -> None:
    def _san(obj):
        if isinstance(obj, dict):
            return {k: _san(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_san(x) for x in obj]
        if isinstance(obj, (float, np.floating, int)):
            return _finite(float(obj))
        return obj
    data = _san(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sørger for at der findes en kolonne 'timestamp' i UTC (tz-naiv).
    Henter fra en række mulige felter eller fra index.
    """
    out = df.copy()
    src = None
    for c in ("timestamp","datetime","ts","time","date","open_time","close_time","openTime","closeTime"):
        if c in out.columns:
            src = out[c]; break
    if src is None:
        if isinstance(out.index, pd.DatetimeIndex):
            src = out.index.to_series(index=out.index, name="timestamp")
        else:
            return out  # caller håndterer

    try:
        if np.issubdtype(src.dtype, np.number):
            unit = "ms" if float(src.iloc[-1]) > 1e11 else "s"
            ts = pd.to_datetime(src, unit=unit, utc=True)
        else:
            ts = pd.to_datetime(src, utc=True, errors="coerce")
        out["timestamp"] = ts.dt.tz_convert(None)
    except Exception:
        out["timestamp"] = pd.to_datetime(src, errors="coerce")
    return out

def _map_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Map ikke-numeriske features til tal (f.eks. regime)."""
    out = df.copy()
    if "regime" in out.columns and not np.issubdtype(out["regime"].dtype, np.number):
        regime_map = {"bull": 1, "neutral": 0, "bear": -1, "trend": 1, "meanrev": -1}
        out["regime"] = out["regime"].map(regime_map).fillna(0).astype(float)
    return out

def _build_model_frame(feats_all: pd.DataFrame, feature_order: Optional[List[str]]) -> pd.DataFrame:
    """
    Returnér en DataFrame med nøjagtigt de kolonner modellen forventer.
    - Hvis feature_order findes, bruger vi den (udfylder manglende med 0.0).
    - Ellers: brug alle numeriske kolonner undtagen 'timestamp'.
    """
    df = _map_non_numeric(feats_all.copy())

    if feature_order and len(feature_order) > 0:
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0.0
        X = df[feature_order].copy()
    else:
        cand = [c for c in df.columns if c != "timestamp"]
        num = [c for c in cand if np.issubdtype(df[c].dtype, np.number)]
        X = df[num].copy()

    # Drop NaN i features
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

def _ensure_daily_metrics_headers() -> None:
    if not DAILY_METRICS_CSV.exists():
        _append_csv(
            DAILY_METRICS_CSV,
            ["date","signal_count","trades","win_rate","gross_pnl","net_pnl","max_dd","sharpe_d"],
            []
        )

def _calc_and_upsert_daily_metrics(date_str: str) -> None:
    """Simpel dagsmetrik – robust mod tomme filer. (FutureWarning-fri concat)"""
    _ensure_daily_metrics_headers()

    gross, wins, closed, commissions = 0.0, 0, 0, 0.0
    if FILLS_CSV.exists():
        try:
            df = pd.read_csv(FILLS_CSV)
            if "ts" in df.columns:
                df["date"] = df["ts"].astype(str).str[:10]
                d = df[df["date"] == date_str].copy()
                if not d.empty:
                    if "pnl_realized" in d.columns:
                        pnl = pd.to_numeric(d["pnl_realized"], errors="coerce").fillna(0.0)
                        gross = float(pnl.sum())
                        wins = int((pnl > 0).sum())
                        closed = int((pnl != 0).sum())
                    if "commission" in d.columns:
                        commissions = float(pd.to_numeric(d["commission"], errors="coerce").fillna(0.0).sum())
        except Exception:
            pass

    if closed == 0 and SIGNALS_CSV.exists():
        try:
            s = pd.read_csv(SIGNALS_CSV)
            if {"ts","signal"}.issubset(s.columns):
                s["date"] = s["ts"].astype(str).str[:10]
                d = s[s["date"] == date_str].copy().sort_values("ts")
                sig = pd.to_numeric(d["signal"], errors="coerce").fillna(0).astype(int).to_numpy()
                closed = int(((sig[:-1] == 1) & (sig[1:] == 0)).sum())
        except Exception:
            pass

    net = gross - commissions
    max_dd = 0.0
    sharpe_d = 0.0
    if EQUITY_CSV.exists():
        try:
            e = pd.read_csv(EQUITY_CSV)
            if {"date","equity"}.issubset(e.columns):
                day = e[e["date"].astype(str) == date_str]
                arr = pd.to_numeric(day["equity"], errors="coerce").dropna().values
                if arr.size:
                    peak = -1e18
                    dd_pct = 0.0
                    for v in arr:
                        peak = max(peak, v)
                        dd_pct = min(dd_pct, (v - peak) / (peak + 1e-12) * 100.0)
                    max_dd = float(dd_pct)
                    rets = np.diff(arr)
                    if rets.size > 1 and np.std(rets) > 1e-12:
                        sharpe_d = float(np.mean(rets) / np.std(rets))
        except Exception:
            pass

    signal_count = 0
    if SIGNALS_CSV.exists():
        try:
            s = pd.read_csv(SIGNALS_CSV)
            s["date"] = s["ts"].astype(str).str[:10]
            signal_count = int((s["date"] == date_str).sum())
        except Exception:
            pass

    win_rate = float(wins / max(closed, 1) * 100.0)

    # læs eksisterende DM
    try:
        dm = pd.read_csv(DAILY_METRICS_CSV)
    except Exception:
        dm = pd.DataFrame(columns=["date","signal_count","trades","win_rate","gross_pnl","net_pnl","max_dd","sharpe_d"])

    cols = ["date","signal_count","trades","win_rate","gross_pnl","net_pnl","max_dd","sharpe_d"]
    row = {
        "date": str(date_str),
        "signal_count": int(signal_count),
        "trades": int(closed),
        "win_rate": round(win_rate, 2),
        "gross_pnl": round(_finite(gross), 2),
        "net_pnl": round(_finite(net), 2),
        "max_dd": round(_finite(max_dd), 2),
        "sharpe_d": round(_finite(sharpe_d), 2),
    }
    new_row_df = pd.DataFrame([row], columns=cols)
    if dm.empty:
        dm = new_row_df
    elif (dm["date"] == date_str).any():
        dm.loc[dm["date"] == date_str, cols] = new_row_df.iloc[0].values
    else:
        dm = pd.concat([dm, new_row_df], ignore_index=True)

    dm.to_csv(DAILY_METRICS_CSV, index=False)

def _load_last_processed_ts() -> Optional[pd.Timestamp]:
    if LAST_TS_FILE.exists():
        try:
            s = LAST_TS_FILE.read_text(encoding="utf-8").strip()
            if s:
                return pd.to_datetime(s)
        except Exception:
            pass
    return None

def _save_last_processed_ts(ts: pd.Timestamp) -> None:
    LAST_TS_FILE.write_text(ts.isoformat(), encoding="utf-8")


# =========================
# Main loop
# =========================
def main():
    load_dotenv()

    exchange = os.getenv("EXCHANGE", "binance")
    symbol_slashed = os.getenv("SYMBOL", "BTC/USDT")
    symbol_log = symbol_slashed.replace("/", "")
    timeframe = os.getenv("TIMEFRAME", "1h")
    mode_live = (os.getenv("LIVE_MODE", "paper").lower() == "live")

    threshold = float(os.getenv("ENSEMBLE_THRESHOLD", "0.6") or 0.6)
    sleep_s = float(os.getenv("LIVE_POLL_SECS", "30") or 30)

    device = "cpu"
    model_p  = Path(os.getenv("MODEL_PATH", ROOT / "models/best_pytorch_model.pt"))
    scaler_p = Path(os.getenv("SCALER_PATH", ROOT / "models/scaler.pkl"))
    order_p  = Path(os.getenv("FEATURE_ORDER", ROOT / "models/feature_order.json"))

    model  = load_torch_model(model_p, device=device)
    scaler = load_scaler(scaler_p)
    f_order= load_feature_order(order_p)  # kan være None

    if mode_live:
        # Lazily importér først hvis nødvendigt (bevar fremtidig mulighed for B)
        from bot.brokers.ccxt_broker import CcxtBroker  # type: ignore
        api_key  = os.getenv("BINANCE_API_KEY") or None
        secret   = os.getenv("BINANCE_API_SECRET") or None
        broker = CcxtBroker(exchange, api_key, secret, True)
        print("[LIVE] Kører i LIVE-mode (pas på!).")
    else:
        broker = PaperBroker(logs_dir=LOGS, symbol=symbol_slashed)
        print("[LIVE] Kører i PAPER-mode – skriver til logs/*")

    last_done = _load_last_processed_ts()
    if last_done:
        print(f"[LIVE] Sidst behandlede bar: {last_done}")

    try:
        while True:
            try:
                # 1) OHLCV
                df_raw = fetch_ohlcv_df(
                    exchange, symbol_slashed, timeframe, limit=600,  # ekstra warmup
                    api_key=None, secret=None
                )
                if df_raw.empty:
                    print("[LIVE] Tom data – prøver igen.")
                    time.sleep(sleep_s); continue

                # 2) Features + timestamp-normalisering
                feats_all = compute_features(df_raw)
                feats_all = _ensure_timestamp_column(feats_all)

                if feats_all.empty or "timestamp" not in feats_all.columns:
                    print("[LIVE] Ingen gyldige features/timestamp – prøver igen.")
                    time.sleep(sleep_s); continue

                # senest lukket bar
                bar_ts = pd.to_datetime(feats_all["timestamp"].iloc[-1])

                if last_done is not None and bar_ts <= last_done:
                    time.sleep(sleep_s); continue

                price = float(feats_all["close"].iloc[-1])

                # 3) Forbered feature-matrix til modellen
                X_df = _build_model_frame(feats_all, f_order)
                want_dim = len(f_order) if f_order else X_df.shape[1]
                if X_df.shape[1] != want_dim:
                    print(f"[LIVE][warn] feature-dim={X_df.shape[1]} forventet={want_dim}. "
                          f"eksempel-kolonner={list(X_df.columns)[:8]}")

                # 4) Inference – giv X_df og dens kolonne-orden videre
                pred, prob = run_inference(X_df, list(X_df.columns), scaler, model)
                last_sig   = int(pred[-1])  # 1=BUY, 0=SELL
                last_prob  = float(prob[-1]) if isinstance(prob, (list, np.ndarray, pd.Series)) else float(prob)

                regime = "trend"
                if "ema_200" in feats_all.columns:
                    regime = "trend" if float(feats_all["close"].iloc[-1]) >= float(feats_all["ema_200"].iloc[-1]) else "meanrev"
                side_txt = "BUY" if last_sig == 1 else "SELL"

                print(f"[LIVE] {bar_ts}  signal={last_sig} ({side_txt})  prob={last_prob:.3f}  price={price}  (features={X_df.shape[1]})")

                # 5) Log signal til CSV + JSON (til GUI)
                _append_csv(SIGNALS_CSV, ["ts","signal"], [[_ts_iso(bar_ts), last_sig]])
                try:
                    hist = []
                    if SIM_SIGNALS_JSON.exists():
                        try:
                            hist = json.loads(SIM_SIGNALS_JSON.read_text(encoding="utf-8"))
                            if not isinstance(hist, list):
                                hist = []
                        except Exception:
                            hist = []
                    hist.append({
                        "ts": _ts_iso(bar_ts),
                        "symbol": symbol_log,
                        "side": side_txt,
                        "confidence": round(_finite(last_prob), 3),
                        "price": round(_finite(price), 2),
                        "sl": 0.0, "tp": 0.0,
                        "regime": regime,
                    })
                    hist = hist[-200:]
                    _write_json_list_safely(SIM_SIGNALS_JSON, hist)
                except Exception as e:
                    print(f"[LIVE] Kunne ikke skrive sim_signals.json: {e}")

                # 6) Execution
                #    - Vi kalder altid PaperBroker for at få MTM-snapshot pr. bar.
                #    - Vi åbner/lukker kun når prob >= threshold; ellers holder vi nuværende tilstand.
                ts_iso = pd.Timestamp(bar_ts).tz_localize("UTC").tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S")
                eff_sig = last_sig
                if not mode_live:
                    st = broker.status()  # {"mode": "FLAT"/"LONG", ...}
                    if last_prob < threshold:
                        eff_sig = 1 if st.get("mode") == "LONG" else 0
                    # PaperBroker håndterer selv, om der faktisk åbnes/lukkes
                    broker.exec_signal(signal=int(eff_sig), price=float(price), ts=ts_iso)
                else:
                    # (Fremtidig B): CCXT execution (ikke del af A)
                    if last_prob >= threshold:
                        # Implementér rigtigt ordreflow her ved skift til B/testnet
                        pass

                # 7) Dagsmetrikker + persist TS
                _calc_and_upsert_daily_metrics(bar_ts.strftime("%Y-%m-%d"))
                _save_last_processed_ts(bar_ts)

            except Exception as e:
                print(f"[LIVE] Fejl: {e}")

            if sleep_s <= 0:
                break
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("[LIVE] Stoppet af bruger.")


if __name__ == "__main__":
    main()
