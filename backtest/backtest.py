# backtest/backtest.py
import argparse
import datetime
import os
import subprocess

import numpy as np
import pandas as pd

from ensemble.ensemble_predict import ensemble_predict
from strategies.advanced_strategies import (add_adaptive_sl_tp,
                                            ema_crossover_strategy,
                                            ema_rsi_adx_strategy,
                                            ema_rsi_regime_strategy,
                                            regime_ensemble,
                                            rsi_mean_reversion,
                                            voting_ensemble)
from strategies.ema_cross_strategy import ema_cross_signals
from strategies.gridsearch_strategies import grid_search_sl_tp_ema
from strategies.macd_strategy import macd_cross_signals
from strategies.rsi_strategy import rsi_rule_based_signals
from utils.file_utils import save_with_metadata
from utils.log_utils import log_device_status
from utils.metrics_utils import advanced_performance_metrics
from utils.project_path import PROJECT_ROOT
from utils.robust_utils import safe_run
from utils.telegram_utils import send_image, send_message

# === Monitoring-parametre fra config ===
try:
    from config.monitoring_config import (ALARM_THRESHOLDS, ALERT_ON_DRAWNDOWN,
                                          ALERT_ON_PROFIT, ALERT_ON_WINRATE,
                                          ENABLE_MONITORING)
except ImportError:
    ALARM_THRESHOLDS = {"drawdown": -20, "winrate": 0.20, "profit": -10}
    ALERT_ON_DRAWNDOWN = True
    ALERT_ON_WINRATE = True
    ALERT_ON_PROFIT = True
    ENABLE_MONITORING = True

from utils.monitoring_utils import send_live_metrics

FORCE_DEBUG = False
FORCE_DUMMY_TRADES = False

# Simple model: investér hele balancen ved entry, flad når ingen position
FEE = 0.0004
SL_PCT = 0.006
TP_PCT = 0.012
ALLOW_SHORT = True


# ------------------------------
# Hjælpefunktioner
# ------------------------------
def _to_datetime(s):
    """Robust konvertering til datetime (accepterer str/int/np.datetime64)."""
    try:
        if np.issubdtype(pd.Series(s).dtype, np.number):
            return pd.to_datetime(s, unit="s", errors="coerce")
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def walk_forward_splits(df, train_size=0.6, test_size=0.2, step_size=0.1, min_train=20):
    n = len(df)
    train_len = int(n * train_size)
    test_len = int(n * test_size)
    step_len = int(n * step_size)
    start = 0
    splits = []
    while start + train_len + test_len <= n:
        train_idx = np.arange(start, start + train_len)
        test_idx = np.arange(start + train_len, start + train_len + test_len)
        if len(train_idx) >= min_train and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
        start += step_len
    return splits


def compute_regime(
    df: pd.DataFrame, ema_col: str = "ema_200", price_col: str = "close"
) -> pd.DataFrame:
    """Tilføj/returnér 'regime' kolonne (bull/bear/neutral) – med fallback hvis EMA mangler."""
    out = df.copy()
    if "regime" in out.columns:
        return out

    local_ema_col = ema_col
    if local_ema_col not in out.columns:
        if "ema_50" in out.columns:
            local_ema_col = "ema_50"
        else:
            out[ema_col] = out[price_col].ewm(span=200, adjust=False).mean()
            local_ema_col = ema_col

    out["regime"] = np.where(
        out[price_col] > out[local_ema_col],
        "bull",
        np.where(out[price_col] < out[local_ema_col], "bear", "neutral"),
    )
    return out


def regime_filter(signals, regime_col, active_regimes=["bull"]):
    return [
        sig if reg in active_regimes else 0 for sig, reg in zip(signals, regime_col)
    ]


def regime_performance(trades_df, regime_col="regime"):
    if trades_df is None or trades_df.empty or regime_col not in trades_df.columns:
        return {}
    grouped = trades_df.groupby(regime_col)
    results = {}
    for name, group in grouped:
        n = len(group)
        win_rate = (
            (group["profit"] > 0).mean() if n > 0 and "profit" in group.columns else 0.0
        )
        profit_pct = group["profit"].sum() if "profit" in group.columns else 0.0
        drawdown_pct = group["drawdown"].min() if "drawdown" in group.columns else None
        results[name] = {
            "num_trades": int(n),
            "win_rate": float(win_rate),
            "profit_pct": float(profit_pct),
            "drawdown_pct": None if drawdown_pct is None else float(drawdown_pct),
        }
    return results


def get_git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def force_trade_signals(length):
    return np.array([1 if i % 2 == 0 else -1 for i in range(length)], dtype=int)


def clean_signals(signals, length):
    """Sørger for korrekt længde + caster til int. Tillader {1,0} og {1,-1}."""
    if isinstance(signals, pd.Series):
        signals = signals.values
    sig = np.asarray(signals).astype(int)
    if len(sig) != length:
        print(
            f"[ADVARSEL] Signal-længde ({len(sig)}) matcher ikke data-længde ({length}) – padder med 0."
        )
        out = np.zeros(length, dtype=int)
        out[: min(len(sig), length)] = sig[: min(len(sig), length)]
        return out
    return sig


def _interpolate_to(trades_df, x_from_ts, y_vals):
    """Sikker interpolation af drawdown til trades tidsstempler."""
    try:
        x_from = pd.to_datetime(x_from_ts, errors="coerce").astype("int64")
        x_tr = pd.to_datetime(trades_df["timestamp"], errors="coerce").astype("int64")
        return np.interp(x_tr, x_from, y_vals)
    except Exception:
        return np.full(len(trades_df), np.nan)


# ------------------------------
# Backtest
# ------------------------------
def run_backtest(
    df: pd.DataFrame,
    signals: list | np.ndarray | pd.Series | None = None,
    initial_balance: float = 1000.0,
    fee: float = FEE,
    sl_pct: float = SL_PCT,
    tp_pct: float = TP_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simpel long/short/flat-backtest:
    - Signal  1 = long, -1 = short (hvis ALLOW_SHORT=True), 0 = flat.
    - Hele balancen "investeres" ved entry (procent-PnL på konto).
    - Fee fratrækkes ved både entry og exit (multiplicativt).
    - Lukker position ved:
        * Stop-loss / Take-profit
        * Skift til 0 (flat) eller modsat signal
        * Sidste bar
    Returnerer:
      trades_df: kolonner ['timestamp','type','price','balance','regime','profit','drawdown','side']
                 profit i DECIMAL (fx 0.012 = +1.2%)
      balance_df: kolonner ['timestamp','balance','equity','close','drawdown']
    """
    df = df.copy()

    # --- timestamps & kolonne-sikring
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    for col in ("timestamp", "close"):
        if col not in df.columns:
            raise ValueError(
                f"❌ Mangler kolonnen '{col}' i DataFrame til backtest! ({list(df.columns)})"
            )
    df["timestamp"] = _to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp", "close"]).reset_index(drop=True)

    # --- signaler
    if signals is not None:
        df["signal"] = clean_signals(signals, len(df))
    elif "signal" not in df.columns:
        raise ValueError("Ingen signaler angivet til backtest!")
    # Accepter {0,1} → behandle som long/flat
    unique_sig = set(pd.Series(df["signal"]).dropna().unique().tolist())
    if unique_sig.issubset({0, 1}):
        allow_short_here = False
    else:
        allow_short_here = bool(ALLOW_SHORT)

    # --- regime
    df = compute_regime(df)

    trades: list[dict] = []
    balance_logs: list[dict] = []

    balance = float(initial_balance)
    position_open = False
    direction = None  # 'long'/'short'
    entry_price = None
    entry_regime = None

    for i, row in df.iterrows():
        ts = row["timestamp"]
        price = float(row["close"])
        sig = int(row["signal"])
        reg = row.get("regime", "unknown")

        # EXIT: hvis vi har position
        if position_open:
            # Prisbaseret ændring (decimal)
            change = (
                (price - entry_price) / entry_price
                if direction == "long"
                else (entry_price - price) / entry_price
            )

            hit_sl = change <= -sl_pct
            hit_tp = change >= tp_pct
            sig_close = (
                (sig == 0)
                or (direction == "long" and sig < 0)
                or (direction == "short" and sig > 0)
            )

            if hit_sl or hit_tp or sig_close or (i == df.index[-1]):
                # exit-fee
                balance *= 1 - fee
                # realiser PnL
                balance *= 1 + change
                trades.append(
                    {
                        "timestamp": ts,
                        "type": "TP" if hit_tp else ("SL" if hit_sl else "CLOSE"),
                        "price": price,
                        "balance": balance,
                        "regime": entry_regime,
                        "profit": float(change),
                        "drawdown": None,
                        "side": direction,
                    }
                )
                position_open = False
                direction = None
                entry_price = None
                entry_regime = None

        # ENTRY: hvis ingen position
        if not position_open:
            if sig == 1:
                direction = "long"
                entry_price = price
                entry_regime = reg
                position_open = True
                balance *= 1 - fee  # entry-fee
                trades.append(
                    {
                        "timestamp": ts,
                        "type": "BUY",
                        "price": price,
                        "balance": balance,
                        "regime": reg,
                        "profit": 0.0,
                        "drawdown": None,
                        "side": "long",
                    }
                )
            elif sig == -1 and allow_short_here:
                direction = "short"
                entry_price = price
                entry_regime = reg
                position_open = True
                balance *= 1 - fee
                trades.append(
                    {
                        "timestamp": ts,
                        "type": "SELL",
                        "price": price,
                        "balance": balance,
                        "regime": reg,
                        "profit": 0.0,
                        "drawdown": None,
                        "side": "short",
                    }
                )

        # Log balance for hver bar
        balance_logs.append(
            {"timestamp": ts, "balance": balance, "equity": balance, "close": price}
        )

    # Hvis stadig åben på sidste bar (edge-case)
    if position_open:
        ts = df["timestamp"].iloc[-1]
        price = float(df["close"].iloc[-1])
        change = (
            (price - entry_price) / entry_price
            if direction == "long"
            else (entry_price - price) / entry_price
        )
        balance *= 1 - fee
        balance *= 1 + change
        trades.append(
            {
                "timestamp": ts,
                "type": "CLOSE",
                "price": price,
                "balance": balance,
                "regime": entry_regime,
                "profit": float(change),
                "drawdown": None,
                "side": direction,
            }
        )
        balance_logs.append(
            {"timestamp": ts, "balance": balance, "equity": balance, "close": price}
        )

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_logs)

    # --- Drawdown (i %) med clamp til [-100, 0]
    if not balance_df.empty:
        bal = (
            pd.to_numeric(
                balance_df.get("balance", balance_df.get("equity")), errors="coerce"
            )
            .ffill()
            .bfill()
        )
        peak = bal.cummax().replace(0, np.nan)
        dd_pct = ((bal / peak) - 1.0) * 100.0
        balance_df["drawdown"] = dd_pct.fillna(0.0).clip(lower=-100.0, upper=0.0)
        # Interpolér drawdown ind i trades_df
        if not trades_df.empty:
            trades_df["drawdown"] = _interpolate_to(
                trades_df, balance_df["timestamp"], balance_df["drawdown"].values
            )

    # --- Kolonne-sikring & typer
    for col in [
        "timestamp",
        "type",
        "price",
        "balance",
        "regime",
        "profit",
        "drawdown",
        "side",
    ]:
        if col not in trades_df.columns:
            trades_df[col] = np.nan
    if not trades_df.empty:
        trades_df["timestamp"] = _to_datetime(trades_df["timestamp"])

    for col in ["timestamp", "balance", "equity", "close", "drawdown"]:
        if col not in balance_df.columns:
            balance_df[col] = np.nan
    if not balance_df.empty:
        balance_df["timestamp"] = _to_datetime(balance_df["timestamp"])
        # sørg for 'balance' findes (alias fra equity)
        if "balance" not in balance_df.columns or balance_df["balance"].isna().all():
            if "equity" in balance_df.columns:
                balance_df["balance"] = pd.to_numeric(
                    balance_df["equity"], errors="coerce"
                )

    # Debug prints
    print("TRADES DF (head):\n", trades_df.head())
    print("BALANCE DF (head):\n", balance_df.head())
    if len(trades_df) < 2:
        print("‼️ ADVARSEL: Få eller ingen handler genereret!")

    # === Dummy fallback til test/debug ===
    if FORCE_DUMMY_TRADES and (trades_df.empty or trades_df.shape[0] < 2):
        print("‼️ FORCE DEBUG: Genererer dummy trades for test!")
        trades_df = pd.DataFrame(
            [
                {
                    "timestamp": df.iloc[0]["timestamp"],
                    "type": "BUY",
                    "price": df.iloc[0]["close"],
                    "balance": initial_balance,
                    "regime": df.iloc[0]["regime"],
                    "profit": 0.0,
                    "drawdown": None,
                    "side": "long",
                },
                {
                    "timestamp": df.iloc[-1]["timestamp"],
                    "type": "TP",
                    "price": df.iloc[-1]["close"],
                    "balance": initial_balance * 1.1,
                    "regime": df.iloc[-1]["regime"],
                    "profit": 0.1,
                    "drawdown": None,
                    "side": "long",
                },
                {
                    "timestamp": df.iloc[-1]["timestamp"],
                    "type": "SL",
                    "price": df.iloc[-1]["close"],
                    "balance": initial_balance * 1.05,
                    "regime": df.iloc[-1]["regime"],
                    "profit": -0.05,
                    "drawdown": None,
                    "side": "long",
                },
            ]
        )
        balance_df = pd.DataFrame(
            [
                {
                    "timestamp": df.iloc[0]["timestamp"],
                    "balance": initial_balance,
                    "equity": initial_balance,
                    "close": df.iloc[0]["close"],
                },
                {
                    "timestamp": df.iloc[-1]["timestamp"],
                    "balance": initial_balance * 1.1,
                    "equity": initial_balance * 1.1,
                    "close": df.iloc[-1]["close"],
                },
                {
                    "timestamp": df.iloc[-1]["timestamp"],
                    "balance": initial_balance * 1.05,
                    "equity": initial_balance * 1.05,
                    "close": df.iloc[-1]["close"],
                },
            ]
        )
        # genberegn drawdown
        bal = pd.to_numeric(balance_df["balance"], errors="coerce")
        dd = ((bal / bal.cummax().replace(0, np.nan)) - 1.0) * 100.0
        balance_df["drawdown"] = dd.fillna(0.0).clip(lower=-100.0, upper=0.0)
        trades_df["drawdown"] = _interpolate_to(
            trades_df, balance_df["timestamp"], balance_df["drawdown"].values
        )

    return trades_df, balance_df


# ------------------------------
# Metrics & persist
# ------------------------------
def calc_backtest_metrics(trades_df, balance_df, initial_balance=1000.0):
    """
    Wrapper der kalder advanced_performance_metrics og normaliserer output.
    """
    try:
        metrics = advanced_performance_metrics(trades_df, balance_df, initial_balance)
    except Exception as e:
        print(f"[WARN] advanced_performance_metrics fejlede: {e}")
        # simpel fallback: fra første/sidste balance
        if balance_df is None or balance_df.empty:
            return {
                "profit_pct": 0.0,
                "win_rate": 0.0,
                "drawdown_pct": 0.0,
                "num_trades": 0,
                "max_consec_losses": 0,
                "recovery_bars": -1,
                "profit_factor": "N/A",
                "sharpe": 0.0,
                "sortino": 0.0,
            }
        col = "balance" if "balance" in balance_df.columns else "equity"
        bal = pd.to_numeric(balance_df[col], errors="coerce").dropna()
        if bal.size < 2:
            prof = 0.0
        else:
            prof = (bal.iloc[-1] / max(bal.iloc[0], 1e-9) - 1.0) * 100.0
        dd_series = (bal / bal.cummax() - 1.0) * 100.0
        dd_min = float(dd_series.min()) if len(dd_series) else 0.0
        return {
            "profit_pct": float(prof),
            "win_rate": 0.0,
            "drawdown_pct": float(max(dd_min, -100.0)),
            "num_trades": (
                int(len(trades_df[trades_df["type"].isin(["TP", "SL", "CLOSE"])]))
                if trades_df is not None
                else 0
            ),
            "max_consec_losses": 0,
            "recovery_bars": -1,
            "profit_factor": "N/A",
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    out = {
        "profit_pct": float(metrics.get("profit_pct", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "drawdown_pct": float(
            metrics.get("max_drawdown", metrics.get("drawdown_pct", 0.0))
        ),
        "num_trades": (
            int(len(trades_df[trades_df["type"].isin(["TP", "SL", "CLOSE"])]))
            if trades_df is not None
            else 0
        ),
        "max_consec_losses": int(metrics.get("max_consec_losses", 0) or 0),
        "recovery_bars": int(metrics.get("recovery_bars", -1) or -1),
        "profit_factor": metrics.get("profit_factor", "N/A"),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "sortino": float(metrics.get("sortino", 0.0)),
    }
    return out


def save_backtest_results(
    metrics, version="v1", csv_path=PROJECT_ROOT / "data" / "backtest_results.csv"
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()
    row = {"timestamp": timestamp, "version": version, "git_hash": git_hash, **metrics}
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    print(f"✅ Backtest-metrics logget til: {csv_path}")


# ------------------------------
# CLI & main
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_path",
        type=str,
        default=PROJECT_ROOT
        / "outputs"
        / "feature_data/btc_1h_features_v_test_20250610.csv",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=PROJECT_ROOT / "data" / "backtest_results.csv",
    )
    parser.add_argument(
        "--balance_path", type=str, default=PROJECT_ROOT / "data" / "balance.csv"
    )
    parser.add_argument(
        "--trades_path", type=str, default=PROJECT_ROOT / "data" / "trades.csv"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ensemble",
        choices=["ensemble", "voting", "regime", "ema_rsi", "meanrev"],
    )
    parser.add_argument("--gridsearch", action="store_true")
    parser.add_argument(
        "--voting",
        type=str,
        default="majority",
        choices=["majority", "weighted", "sum"],
    )
    parser.add_argument("--debug_ensemble", action="store_true")
    parser.add_argument(
        "--walkforward", action="store_true", help="Aktiver walk-forward analyse"
    )
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument(
        "--force_trades",
        action="store_true",
        help="Tving signaler til BUY/SELL for test/debug",
    )
    return parser.parse_args()


def load_csv_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if str(first_line).startswith("#"):
        print("[INFO] Meta-header fundet i CSV – loader med skiprows=1")
        return pd.read_csv(file_path, skiprows=1)
    else:
        return pd.read_csv(file_path)


def main():
    args = parse_args()
    log_device_status(
        context="backtest",
        extra={"strategy": args.strategy, "feature_file": str(args.feature_path)},
        telegram_func=send_message,
    )

    df = load_csv_auto(args.feature_path)
    if "datetime" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    print("Indlæst data med kolonner:", list(df.columns))
    df = compute_regime(df)

    if args.gridsearch:
        print("🔍 Kører gridsearch for EMA-strategier...")
        results_df = grid_search_sl_tp_ema(
            df,
            sl_grid=np.linspace(0.003, 0.012, 5),
            tp_grid=np.linspace(0.008, 0.025, 5),
            ema_fast_grid=[7, 9, 12, 15],
            ema_slow_grid=[21, 30, 34, 55],
            regime_only=False,
            top_n=10,
            log_path=PROJECT_ROOT / "outputs" / "gridsearch/ema_gridsearch_results.csv",
        )
        print(results_df.head(10))
        return

    # === WALK-FORWARD ANALYSE ===
    if args.walkforward:
        splits = walk_forward_splits(
            df,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
        )
        all_metrics = []
        for i, (train_idx, test_idx) in enumerate(splits):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx].copy()
            if args.force_trades:
                signals = np.random.choice([1, -1], size=len(df_test))
            else:
                ml_signals = rsi_rule_based_signals(df_test, low=35, high=65)
                rsi_signals = rsi_rule_based_signals(df_test, low=40, high=60)
                macd_signals = macd_cross_signals(df_test)
                ema_signals = ema_cross_signals(df_test)
                signals = ensemble_predict(
                    ml_signals,
                    rsi_signals,
                    rule_preds=macd_signals,
                    extra_preds=[ema_signals],
                    voting=args.voting,
                    debug=args.debug_ensemble,
                )
            sig_dist = pd.Series(signals).value_counts().to_dict()
            print(f"Signal dist vindue {i+1}: {sig_dist}")
            try:
                send_message(f"Signal dist vindue {i+1}: {sig_dist}")
            except Exception:
                pass
            df_test["signal"] = signals
            trades_df, balance_df = run_backtest(df_test, signals=signals)
            metrics = calc_backtest_metrics(trades_df, balance_df)
            all_metrics.append(metrics)
            summary = f"Walk-forward vindue {i+1}:\n" + "\n".join(
                [f"{k}: {v}" for k, v in metrics.items()]
            )
            try:
                if trades_df.empty:
                    send_message(
                        f"‼️ Ingen handler i vindue {i+1}! Signal dist: {sig_dist}"
                    )
                else:
                    send_message(summary)
            except Exception:
                pass
            # Monitoring per vindue
            if ENABLE_MONITORING:
                try:
                    send_live_metrics(
                        trades_df,
                        balance_df,
                        symbol="WALKFORWARD",
                        timeframe=f"win{i+1}",
                        thresholds=ALARM_THRESHOLDS,
                        alert_on_drawdown=ALERT_ON_DRAWNDOWN,
                        alert_on_winrate=ALERT_ON_WINRATE,
                        alert_on_profit=ALERT_ON_PROFIT,
                    )
                except Exception:
                    pass
        metrics_df = pd.DataFrame(all_metrics)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs("outputs", exist_ok=True)
            metrics_df[["profit_pct", "drawdown_pct"]].plot(marker="o")
            plt.title("Walk-forward performance")
            plt.ylabel("Value")
            plt.xlabel("Vindue")
            plt.grid(True)
            plt.tight_layout()
            out_path = PROJECT_ROOT / "outputs" / "walkforward_performance.png"
            plt.savefig(out_path)
            send_image(out_path, caption="📈 Walk-forward analyse")
        except Exception as e:
            print(f"❌ Plot-fejl: {e}")
        return

    # === Fuldt run ===
    if FORCE_DEBUG or args.force_trades:
        print("‼️ DEBUG: Forcerer skiftevis BUY/SELL på hele datasættet")
        signals = force_trade_signals(len(df))
    else:
        if args.strategy == "ensemble":
            ml_signals = rsi_rule_based_signals(df, low=35, high=65)
            rsi_signals = rsi_rule_based_signals(df, low=40, high=60)
            macd_signals = macd_cross_signals(df)
            ema_signals = ema_cross_signals(df)
            signals = ensemble_predict(
                ml_signals,
                rsi_signals,
                rule_preds=macd_signals,
                extra_preds=[ema_signals],
                voting=args.voting,
                debug=args.debug_ensemble,
            )
        elif args.strategy == "voting":
            signals = voting_ensemble(df)["signal"]
        elif args.strategy == "regime":
            signals = regime_ensemble(df)["signal"]
        elif args.strategy == "ema_rsi":
            signals = ema_rsi_regime_strategy(df)["signal"]
        elif args.strategy == "meanrev":
            signals = rsi_mean_reversion(df)["signal"]
        else:
            signals = rsi_rule_based_signals(df)
        print("Signal distribution:", pd.Series(signals).value_counts().to_dict())

    df["signal"] = clean_signals(signals, len(df))
    trades_df, balance_df = run_backtest(df, signals=df["signal"].values)
    metrics = calc_backtest_metrics(trades_df, balance_df)
    save_backtest_results(metrics, version="v1.7.0", csv_path=args.results_path)
    print("Backtest-metrics:", metrics)
    save_with_metadata(balance_df, args.balance_path, version="v1.7.0")
    save_with_metadata(trades_df, args.trades_path, version="v1.7.0")

    reg_stats = regime_performance(trades_df)
    print("Regime performance:", reg_stats)
    try:
        send_message(f"📊 Regime-performance:\n{reg_stats}")
        if ENABLE_MONITORING:
            send_live_metrics(
                trades_df,
                balance_df,
                symbol="BACKTEST",
                timeframe="all",
                thresholds=ALARM_THRESHOLDS,
                alert_on_drawdown=ALERT_ON_DRAWNDOWN,
                alert_on_winrate=ALERT_ON_WINRATE,
                alert_on_profit=ALERT_ON_PROFIT,
            )
    except Exception as e:
        print(f"⚠️ Telegram-fejl: {e}")


if __name__ == "__main__":
    safe_run(main)
