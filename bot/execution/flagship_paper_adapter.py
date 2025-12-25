# bot/execution/flagship_paper_adapter.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.strategies.flagship_trend_v1 import (
    FlagshipTrendConfig,
    FlagshipTrendV1Strategy,
    Signal,
)
from features.auto_features import ensure_latest


@dataclass
class PaperTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str  # "LONG" / "SHORT" (Flagship v1 er LONG-only, men vi holder feltet generisk)
    qty: float
    ret_pct: float  # afkast i %
    pnl_abs: float  # afkast i kontoenheder


def _ensure_timestamp(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "datetime" in df.columns:
        ts_col = "datetime"
        df = df.rename(columns={"datetime": "timestamp"})
    else:
        raise ValueError(f"Mangler timestamp/datetime-kolonne i features: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df, "timestamp"


def _load_flagship_features(symbol: str, interval: str, min_rows: int = 200) -> pd.DataFrame:
    """
    Henter AUTO-feature-fil til Flagship Trend v1.
    Antar at ensure_latest har genereret standard feature-sættet
    (inkl. ema_21, ema_50, rsi_14, atr_14, pv_ratio mv.).
    """
    path = ensure_latest(symbol=symbol, timeframe=interval, min_rows=min_rows)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Features-fil findes ikke: {p}")

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    df, _ = _ensure_timestamp(df)

    required = {"close", "high", "low", "volume", "ema_21", "ema_50", "rsi_14", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Features mangler nødvendige kolonner til Flagship: {sorted(missing)}")

    # pv_ratio kan være optional – hvis den mangler, sætter vi 1.0
    if "pv_ratio" not in df.columns:
        df["pv_ratio"] = 1.0

    return df


def _compute_basic_metrics(trades: List[PaperTrade], equity_series: pd.Series) -> Dict[str, float]:
    """
    Udregner et sæt metrics, der matcher backtest-formatet nogenlunde.
    """
    if equity_series is None or equity_series.empty:
        return {
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "max_consec_losses": 0,
            "recovery_bars": -1,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "num_trades": 0,
        }

    start_eq = float(equity_series.iloc[0])
    end_eq = float(equity_series.iloc[-1])
    profit_pct = (end_eq / max(start_eq, 1e-9) - 1.0) * 100.0

    # Max drawdown
    eq = equity_series.to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    max_dd = float(dd.min()) if dd.size else 0.0

    if not trades:
        return {
            "profit_pct": float(profit_pct),
            "max_drawdown": float(max_dd),
            "win_rate": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "max_consec_losses": 0,
            "recovery_bars": -1,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "num_trades": 0,
        }

    rets = np.array([t.ret_pct / 100.0 for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    win_rate = float((rets > 0).mean() * 100.0)
    best_trade = float(rets.max() * 100.0)
    worst_trade = float(rets.min() * 100.0)

    max_consec_losses = 0
    cur_losses = 0
    for r in rets:
        if r < 0:
            cur_losses += 1
            max_consec_losses = max(max_consec_losses, cur_losses)
        else:
            cur_losses = 0

    sum_win = float(wins.sum()) if wins.size else 0.0
    sum_loss = float(np.abs(losses.sum())) if losses.size else 0.0
    profit_factor = float(sum_win / sum_loss) if sum_loss > 0 else 0.0

    if rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(len(rets)))
    else:
        sharpe = 0.0

    neg = rets[rets < 0]
    if neg.size > 0 and neg.std(ddof=1) > 0:
        sortino = float(rets.mean() / neg.std(ddof=1) * np.sqrt(len(rets)))
    else:
        sortino = 0.0

    return {
        "profit_pct": float(profit_pct),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "best_trade": float(best_trade),
        "worst_trade": float(worst_trade),
        "max_consec_losses": int(max_consec_losses),
        "recovery_bars": -1,
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "num_trades": int(len(trades)),
    }


def run_flagship_paper(
    symbol: str,
    interval: str,
    *,
    tag: str = "v1",
    start_equity: float = 100_000.0,
    risk_per_trade_pct: float = 0.01,
    daily_loss_limit_pct: float = 0.03,
    cooldown_bars_after_exit: int = 3,
    min_rows: int = 200,
) -> Dict[str, float]:
    """
    Paper/pseudo-live kørsel for FlagshipTrendV1.
    - Bruger AUTO-features.
    - Kører bar-for-bar med samme on_bar-API som i unit tests.
    - Simulerer fill-logik direkte i adapteren (ingen ekstern broker).

    Returnerer metrics-dict og skriver CSV/JSON til outputs/paper/.
    """
    df = _load_flagship_features(symbol=symbol, interval=interval, min_rows=min_rows)

    strat_cfg = FlagshipTrendConfig(
        symbol=symbol,
        risk_per_trade_pct=risk_per_trade_pct,
        daily_loss_limit_pct=daily_loss_limit_pct,
        cooldown_bars_after_exit=cooldown_bars_after_exit,
    )
    strat = FlagshipTrendV1Strategy(strat_cfg)

    equity = start_equity
    cash = start_equity
    position_qty = 0.0
    position_side: Optional[str] = None
    entry_price: Optional[float] = None

    trades: List[PaperTrade] = []
    equity_rows: List[Dict[str, object]] = []
    daily_pnl: Dict[str, float] = {}

    for _, row in df.iterrows():
        ts = row["timestamp"]
        date_key = ts.date().isoformat()
        price = float(row["close"])

        bar = {
            "close": price,
            "high": float(row["high"]),
            "low": float(row["low"]),
            "volume": float(row["volume"]),
            "ema_21": float(row["ema_21"]),
            "ema_50": float(row["ema_50"]),
            "rsi_14": float(row["rsi_14"]),
            "atr_14": float(row["atr_14"]),
            "pv_ratio": float(row.get("pv_ratio", 1.0)),
        }

        account_state = {
            "equity": float(equity),
            "position_qty": float(position_qty),
            "position_side": position_side,
            "entry_price": float(entry_price) if entry_price is not None else None,
        }
        day_pnl = daily_pnl.get(date_key, 0.0)

        sig: Optional[Signal] = strat.on_bar(bar, account_state, daily_pnl=day_pnl)

        if sig is not None and sig.qty and sig.qty > 0:
            if sig.side == "BUY":
                # Åbn / øg long
                qty = float(sig.qty)
                notional = qty * price
                cash -= notional
                if position_qty == 0.0:
                    entry_price = price
                else:
                    # gennemsnitspris hvis vi bygger videre på positionen
                    total_notional_old = position_qty * (entry_price or price)
                    entry_price = (total_notional_old + notional) / max(position_qty + qty, 1e-9)
                position_qty += qty
                position_side = "LONG"

            elif sig.side == "SELL" and position_qty > 0.0:
                # Luk/trim long
                qty = min(float(sig.qty), position_qty)
                notional = qty * price
                cash += notional
                if entry_price is not None:
                    pnl_abs = (price - entry_price) * qty
                    pnl_pct = (price / entry_price - 1.0) * 100.0
                else:
                    pnl_abs = 0.0
                    pnl_pct = 0.0

                daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + pnl_abs

                trades.append(
                    PaperTrade(
                        entry_time=ts,  # vi har ikke historisk entry_ts her; kan udvides senere
                        exit_time=ts,
                        entry_price=float(entry_price or price),
                        exit_price=price,
                        direction="LONG",
                        qty=qty,
                        ret_pct=float(pnl_pct),
                        pnl_abs=float(pnl_abs),
                    )
                )

                position_qty -= qty
                if position_qty <= 1e-8:
                    position_qty = 0.0
                    position_side = None
                    entry_price = None

        equity = cash + position_qty * price

        equity_rows.append(
            {
                "timestamp": ts,
                "price": price,
                "equity": float(equity),
                "position_qty": float(position_qty),
                "position_side": position_side or "",
            }
        )

    equity_df = pd.DataFrame(equity_rows).sort_values("timestamp").reset_index(drop=True)
    metrics = _compute_basic_metrics(trades, equity_df["equity"])

    out_dir = Path("outputs") / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"flagship_{symbol.lower()}_{interval}_{tag}"

    trades_path = out_dir / f"{base}_trades.csv"
    equity_path = out_dir / f"{base}_equity.csv"
    metrics_path = out_dir / f"{base}.json"

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    pd.Series(metrics).to_json(metrics_path)

    print("\n[FLAGSHIP PAPER] Kørsel færdig.")
    print(f"- Trades  (CSV): {trades_path}")
    print(f"- Equity  (CSV): {equity_path}")
    print(f"- Metrics (JSON): {metrics_path}")
    print(f"- Nøgletal: {metrics}")

    return metrics
