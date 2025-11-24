# backtest/walkforward.py
"""
Walk-forward + regime-split modul (EPIC B2).

Formål:
- Køre walk-forward backtests på et eksisterende signal-array.
- Dele OOS-perioder op i volatilitets-regimer.
- Generere en OOS-rapport per regime + per fold og skrive den til outputs/walkforward/.

Afhængigheder:
- backtest.backtest.run_backtest (eksisterende engine).
- utils.metrics_utils.advanced_performance_metrics (valgfrit, fallback hvis ikke tilgængelig).

Primær entrypoint:
    run_walkforward_with_regimes(df, signals, config, symbol, timeframe, out_dir)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.backtest import run_backtest  # type: ignore

# Forsøg at bruge avancerede metrikker, men hav fallback klar
try:
    from utils.metrics_utils import advanced_performance_metrics as _apm  # type: ignore
except Exception:  # pragma: no cover
    _apm = None  # type: ignore


# ---------------------------------------------------------------------------
# Konfiguration & små helpers
# ---------------------------------------------------------------------------


@dataclass
class WalkforwardConfig:
    """Walk-forward konfiguration i antal bars (indeks-baseret)."""

    train_bars: int
    oos_bars: int
    step_bars: Optional[int] = None  # Hvis None → rullende med oos_bars som step
    min_oos_trades: int = 0  # evt. filter på meget tomme OOS-vinduer


@dataclass
class FoldResult:
    fold_idx: int
    train_start: int
    train_end: int
    oos_start: int
    oos_end: int
    profit_pct: float
    max_drawdown: float
    sharpe: float
    num_trades: int


def _simple_metrics_from_balance(
    trades_df: pd.DataFrame, balance_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Minimal fallback-metrik:
    - profit_pct fra første/sidste equity/balance
    - max_drawdown i % ([-100, 0])
    - sharpe fra simple pct-change (annualiseret med sqrt(252))
    """
    try:
        if balance_df is None or balance_df.empty:
            return {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "num_trades": 0}

        col = None
        if "balance" in balance_df.columns:
            col = "balance"
        elif "equity" in balance_df.columns:
            col = "equity"

        if col is None:
            return {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "num_trades": 0}

        series = pd.to_numeric(balance_df[col], errors="coerce").dropna()
        if series.size < 2:
            return {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "num_trades": 0}

        start_val = float(series.iloc[0])
        end_val = float(series.iloc[-1])
        profit_pct = (end_val / max(start_val, 1e-9) - 1.0) * 100.0

        dd_series = (series / series.cummax() - 1.0) * 100.0
        max_dd = float(max(dd_series.min(), -100.0))

        rets = series.pct_change().dropna()
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        num_trades = 0
        if trades_df is not None and not trades_df.empty:
            if "type" in trades_df.columns:
                num_trades = int((trades_df["type"].astype(str).str.upper() == "CLOSE").sum())
            elif "profit" in trades_df.columns:
                num_trades = int((pd.to_numeric(trades_df["profit"], errors="coerce") != 0).sum())

        return {
            "profit_pct": float(profit_pct),
            "max_drawdown": float(max_dd),
            "sharpe": float(sharpe),
            "num_trades": int(num_trades),
        }
    except Exception:
        return {"profit_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "num_trades": 0}


def _compute_metrics(trades_df: pd.DataFrame, balance_df: pd.DataFrame) -> Dict[str, float]:
    """Wrapper som forsøger avancerede metrikker først, derefter fallback."""
    if _apm is not None:
        try:
            m = _apm(trades_df, balance_df)
            # Harmonisér keys
            profit_pct = float(m.get("profit_pct", 0.0))
            max_dd = float(m.get("max_drawdown", m.get("drawdown_pct", 0.0)))
            sharpe = float(m.get("sharpe", 0.0))
            num_trades = int(m.get("num_trades", 0))
            return {
                "profit_pct": profit_pct,
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "num_trades": num_trades,
            }
        except Exception:
            pass
    return _simple_metrics_from_balance(trades_df, balance_df)


# ---------------------------------------------------------------------------
# Regime-detektion
# ---------------------------------------------------------------------------


def assign_volatility_regimes(
    df: pd.DataFrame,
    n_regimes: int = 5,
    vol_window: int = 50,
    close_col: str = "close",
) -> pd.Series:
    """
    Tildel volatilitets-regimer baseret på rullende std af pct-change i close.

    Returnerer:
        pd.Series med heltals-regimer [0 .. n_regimes-1].
    """
    if close_col not in df.columns:
        raise ValueError(f"DataFrame mangler '{close_col}' kolonne til regime-split.")

    price = pd.to_numeric(df[close_col], errors="coerce").ffill().bfill()
    ret = price.pct_change().fillna(0.0)
    vol = ret.rolling(vol_window, min_periods=max(5, vol_window // 5)).std().bfill().fillna(0.0)

    # Primært forsøg: qcut i n_regimes kvantiler
    try:
        q = pd.qcut(vol, q=n_regimes, labels=False, duplicates="drop")
    except Exception:
        q = None

    if q is None or q.isna().all() or q.nunique() < 2:
        # Fallback: cut baseret på lige bins mellem min/max
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax <= vmin:
            return pd.Series(0, index=df.index, name="regime")
        bins = np.linspace(vmin, vmax + 1e-12, n_regimes + 1)
        q = pd.cut(vol, bins=bins, labels=False, include_lowest=True)

    q = q.fillna(0).astype(int)
    q.name = "regime"
    return q


# ---------------------------------------------------------------------------
# Walk-forward slicing
# ---------------------------------------------------------------------------


def _generate_walkforward_slices(
    n_rows: int,
    cfg: WalkforwardConfig,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Genererer (fold_idx, train_start, train_end, oos_start, oos_end) tuples.

    Indeks er [start, end) som i standard Python slicing.
    """
    train = int(cfg.train_bars)
    oos = int(cfg.oos_bars)
    step = int(cfg.step_bars or cfg.oos_bars)

    if train <= 0 or oos <= 0:
        raise ValueError("train_bars og oos_bars skal være > 0.")

    slices: List[Tuple[int, int, int, int, int]] = []
    fold_idx = 0
    start = 0

    while (start + train + oos) <= n_rows:
        train_start = start
        train_end = start + train
        oos_start = train_end
        oos_end = train_end + oos
        slices.append((fold_idx, train_start, train_end, oos_start, oos_end))
        fold_idx += 1
        start += step

    return slices


# ---------------------------------------------------------------------------
# Hovedfunktion: walk-forward + regime-split
# ---------------------------------------------------------------------------


def run_walkforward_with_regimes(
    df: pd.DataFrame,
    signals: np.ndarray,
    cfg: WalkforwardConfig,
    *,
    symbol: str = "UNKNOWN",
    timeframe: str = "1h",
    out_dir: str = "outputs/walkforward",
) -> Dict[str, Any]:
    """
    Kør walk-forward backtest på (df, signals) og generér regime-split OOS-rapport.

    df:
        DataFrame med mindst: ['timestamp', 'close', 'open', 'high', 'low', 'volume'].
    signals:
        np.ndarray, samme længde som df, med binære signaler (0/1 eller -1/0/1).
    cfg:
        WalkforwardConfig

    Returnerer dict med:
        - 'folds': DataFrame med fold-metrics.
        - 'regimes': DataFrame med regime-aggregater.
        - 'folds_path': sti til CSV med fold-resultater.
        - 'regimes_path': sti til CSV med regime-resultater.
        - 'config': cfg som dict.
    """
    if len(df) != len(signals):
        raise ValueError("df og signals SKAL have samme længde. ")

    df = df.copy().reset_index(drop=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # Fallback timestamp, så backtest ikke fejler på tidskolonne
        df["timestamp"] = pd.date_range(start="1970-01-01", periods=len(df), freq="H")

    # Tilføj regimes
    df["regime"] = assign_volatility_regimes(df, n_regimes=5, close_col="close")

    # Slices til walk-forward
    slices = _generate_walkforward_slices(len(df), cfg)
    if not slices:
        raise ValueError("Ingen gyldige walk-forward-slices genereret – check train/oos/step.")

    fold_results: List[FoldResult] = []

    # Aggregation per regime
    regime_acc: Dict[int, Dict[str, List[float]]] = {}

    np_signals = np.asarray(signals)

    for fold_idx, tr_s, tr_e, oos_s, oos_e in slices:
        oos_df = df.iloc[oos_s:oos_e].reset_index(drop=True)
        oos_sig = np_signals[oos_s:oos_e]

        # Eksisterende backtest-engine anvendes på OOS-vinduet
        trades_df, balance_df = run_backtest(oos_df, oos_sig)
        metrics = _compute_metrics(trades_df, balance_df)

        if cfg.min_oos_trades and metrics.get("num_trades", 0) < cfg.min_oos_trades:
            # Skip fold hvis der næsten ingen aktivitet er
            continue

        fr = FoldResult(
            fold_idx=fold_idx,
            train_start=tr_s,
            train_end=tr_e,
            oos_start=oos_s,
            oos_end=oos_e,
            profit_pct=float(metrics.get("profit_pct", 0.0)),
            max_drawdown=float(metrics.get("max_drawdown", 0.0)),
            sharpe=float(metrics.get("sharpe", 0.0)),
            num_trades=int(metrics.get("num_trades", 0)),
        )
        fold_results.append(fr)

        # Regime-split inden for OOS-vinduet
        regimes_in_oos = oos_df["regime"].astype(int).values
        unique_regimes = np.unique(regimes_in_oos)

        for reg in unique_regimes:
            mask = regimes_in_oos == reg
            if not mask.any():
                continue
            # Lige nu tilskrives hele fold-metric til alle regimer der optræder i OOS.
            # Hvis man ønsker mere granularitet, kan man lave sub-backtests per regime.
            acc = regime_acc.setdefault(
                int(reg), {"profit_pct": [], "max_drawdown": [], "sharpe": []}
            )
            acc["profit_pct"].append(fr.profit_pct)
            acc["max_drawdown"].append(fr.max_drawdown)
            acc["sharpe"].append(fr.sharpe)

    # Fold-rapport
    folds_df = (
        pd.DataFrame([asdict(fr) for fr in fold_results])
        if fold_results
        else pd.DataFrame(
            columns=[
                "fold_idx",
                "train_start",
                "train_end",
                "oos_start",
                "oos_end",
                "profit_pct",
                "max_drawdown",
                "sharpe",
                "num_trades",
            ]
        )
    )

    # Regime-rapport
    regime_rows: List[Dict[str, Any]] = []
    for reg, acc in regime_acc.items():
        if not acc["profit_pct"]:
            continue
        regime_rows.append(
            {
                "regime": int(reg),
                "folds": len(acc["profit_pct"]),
                "mean_profit_pct": float(np.mean(acc["profit_pct"])),
                "mean_max_drawdown": float(np.mean(acc["max_drawdown"])),
                "mean_sharpe": float(np.mean(acc["sharpe"])),
            }
        )

    regimes_df = (
        pd.DataFrame(regime_rows).sort_values("regime")
        if regime_rows
        else pd.DataFrame(
            columns=["regime", "folds", "mean_profit_pct", "mean_max_drawdown", "mean_sharpe"]
        )
    )

    # Sørg for at have mindst 5 regimer i rapporten hvis muligt (produktkrav).
    # Hvis der er færre, betyder det reelt at vol-strukturen i datasættet er meget flad.
    # Vi tvinger ikke kunstige regimer frem her – rapporterer blot hvad data giver.

    # Persistér til outputs/
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"walkforward_{symbol}_{timeframe}_{ts_label}"

    folds_path = out_path / f"{base_name}_folds.csv"
    regimes_path = out_path / f"{base_name}_regimes.csv"
    meta_path = out_path / f"{base_name}_meta.json"

    folds_df.to_csv(folds_path, index=False)
    regimes_df.to_csv(regimes_path, index=False)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "config": asdict(cfg),
                "n_rows": len(df),
                "n_folds": int(len(fold_results)),
                "n_regimes": int(regimes_df["regime"].nunique() if not regimes_df.empty else 0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[B2] Walk-forward rapport skrevet til:\n- {folds_path}\n- {regimes_path}\n- {meta_path}"
    )

    return {
        "folds": folds_df,
        "regimes": regimes_df,
        "folds_path": str(folds_path),
        "regimes_path": str(regimes_path),
        "meta_path": str(meta_path),
        "config": asdict(cfg),
    }
