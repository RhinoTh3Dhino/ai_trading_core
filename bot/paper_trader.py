# bot/paper_trader.py

import glob
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # kan fjernes hvis du ikke bruger det senere
import pandas as pd

from config.config import COINS, TIMEFRAMES

# EPIC C – Risk Engine integration
from risk.risk_engine import RiskEngine, RiskLimits, RiskState

# Strategier (voting_ensemble m.fl.)
from strategies.advanced_strategies import (
    add_adaptive_sl_tp,
    ema_crossover_strategy,
    ema_rsi_adx_strategy,
    ema_rsi_regime_strategy,
    voting_ensemble,
)
from utils.performance import print_performance_report
from utils.project_path import PROJECT_ROOT

# -- Parametre (kan importeres fra config.py) --
SL = 0.02
TP = 0.04
START_BALANCE = 10000
FEE = 0.0005

# Standard-risklimits til offline paper trading (kan overrides via argument)
DEFAULT_RISK_LIMITS = RiskLimits(
    max_daily_loss_pct=0.03,
    max_equity_drawdown_pct=0.10,
    max_notional_exposure_pct=0.50,
    max_per_trade_risk_pct=0.01,
    max_open_positions=10,
    circuit_breaker_drawdown_pct=0.05,
    circuit_breaker_cooldown_minutes=60,
)


def find_latest_feature_file(symbol: str, tf: str, version: str = "v1.3") -> str | None:
    """Find seneste feature-fil med mønsteret {symbol}_{tf}_features_{version}_*.csv."""
    pattern = str(
        Path(PROJECT_ROOT)
        / "outputs"
        / "feature_data"
        / f"{symbol.lower()}_{tf}_features_{version}_*.csv"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def plot_trades(df: pd.DataFrame, trades_df: pd.DataFrame, journal_path: Path | str) -> None:
    """
    Plot backtest: pris + køb/salg/SL/TP + risk-blocks.
    Gemmer graf som PNG i samme mappe som journal_path.
    """
    plt.figure(figsize=(14, 7))
    plt.title("Backtest – Signaler & exits")

    # Pris graf
    plt.plot(df["timestamp"], df["close"], label="Pris")

    # Markér køb, salg, SL, TP, risk-blocks
    buys = trades_df[trades_df["type"] == "BUY"]
    sells = trades_df[trades_df["type"] == "SELL"]
    sls = trades_df[trades_df["type"] == "SL"]
    tps = trades_df[trades_df["type"] == "TP"]
    risk_blocks = trades_df[trades_df["type"] == "RISK_BLOCK"]
    force_exits = trades_df[trades_df["type"] == "FORCE_EXIT"]

    if not buys.empty:
        plt.scatter(buys["time"], buys["price"], marker="^", color="green", label="Køb", s=80)
    if not sells.empty:
        plt.scatter(sells["time"], sells["price"], marker="v", color="red", label="Sælg", s=80)
    if not sls.empty:
        plt.scatter(sls["time"], sls["price"], marker="x", color="red", label="Stop Loss", s=80)
    if not tps.empty:
        plt.scatter(
            tps["time"],
            tps["price"],
            marker="*",
            color="gold",
            label="Take Profit",
            s=120,
        )
    if not risk_blocks.empty:
        plt.scatter(
            risk_blocks["time"],
            risk_blocks["price"],
            marker="o",
            color="orange",
            label="Risk block",
            s=70,
        )
    if not force_exits.empty:
        plt.scatter(
            force_exits["time"],
            force_exits["price"],
            marker="D",
            color="blue",
            label="Force exit",
            s=70,
        )

    plt.xlabel("Tid")
    plt.ylabel("Pris")
    plt.legend()
    plt.tight_layout()

    png_path = str(journal_path).replace(".csv", ".png")
    plt.savefig(png_path)
    print(f"✅ Trade-graf gemt som {png_path}")

    plt.show()


def _init_risk_state(start_balance: float, df: pd.DataFrame) -> RiskState:
    """Init RiskState ud fra første timestamp i df eller dagsdato."""
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        try:
            first_ts = pd.to_datetime(df["timestamp"].iloc[0])
            trading_day = first_ts.date()
        except Exception:
            trading_day = datetime.utcnow().date()
    else:
        trading_day = datetime.utcnow().date()

    return RiskState(
        trading_day=trading_day,
        start_equity=start_balance,
        high_equity=start_balance,
        current_equity=start_balance,
        notional_exposure=0.0,
        open_positions=0,
        circuit_breaker_triggered_at=None,
    )


def paper_trade(
    df: pd.DataFrame,
    sl: float = SL,
    tp: float = TP,
    start_balance: float = START_BALANCE,
    fee: float = FEE,
    JOURNAL_PATH: Path | None = None,
    use_adaptive_sl_tp: bool = False,
    enable_risk_checks: bool = True,
    risk_limits: RiskLimits | None = None,
):
    """
    Offline paper backtest med simpel 1-positions-model + RiskEngine integration.

    Forventet input:
      - df['signal'] i {-1, 0, 1}
      - df['close']
      - valgfrit df['timestamp']
      - valgfrit df['sl_pct']/df['tp_pct'] hvis use_adaptive_sl_tp=True

    Logic:
      - Entry ved signal == 1 når der ikke er position.
      - Exit ved signal == -1 eller SL/TP hit.
      - RiskEngine bruges som pre-trade check (kan blokere entry med RISK_BLOCK).
    """
    balance = start_balance
    equity_curve: list[float] = [start_balance]
    position = 0
    entry_price = 0.0
    entry_row = None  # type: ignore[assignment]
    trades: list[dict] = []
    n_wins = 0
    n_trades = 0

    if JOURNAL_PATH is None:
        JOURNAL_PATH = Path(PROJECT_ROOT) / "outputs" / "paper_trades.csv"
    JOURNAL_PATH = Path(JOURNAL_PATH)

    # RiskEngine setup
    if risk_limits is None:
        risk_limits = DEFAULT_RISK_LIMITS
    risk_engine = RiskEngine(risk_limits)
    risk_state = _init_risk_state(start_balance, df)

    # Hoved-loop
    for i, row in df.iterrows():
        ts = row["timestamp"] if "timestamp" in df.columns else i

        # ENTRY
        if position == 0 and row.get("signal", 0) == 1:
            order_price = float(row["close"])
            # simpel model: all-in
            order_notional = balance

            if enable_risk_checks:
                decision = risk_engine.check_pre_trade(
                    risk_state,
                    order_notional=order_notional,
                )
                if not decision.allowed:
                    trades.append(
                        {
                            "time": ts,
                            "type": "RISK_BLOCK",
                            "price": order_price,
                            "balance": balance,
                            "reason": decision.reason,
                            "limit_name": decision.limit_name,
                        }
                    )
                    equity_curve.append(balance)
                    continue

            # Hvis vi når hertil, er ordren godkendt af risk
            position = 1
            entry_price = order_price
            entry_row = row
            trades.append(
                {
                    "time": ts,
                    "type": "BUY",
                    "price": entry_price,
                    "balance": balance,
                }
            )
            n_trades += 1

            # Opdater RiskState for ny eksponering
            risk_state = risk_engine.apply_fill(
                risk_state,
                new_equity=balance,
                new_notional_exposure=order_notional,
                new_open_positions=1,
                now=datetime.utcnow(),
            )

        # EXIT-logik (kun hvis position er åben)
        if position == 1:
            pnl = (float(row["close"]) - entry_price) / entry_price

            if use_adaptive_sl_tp and entry_row is not None:
                this_sl = float(entry_row.get("sl_pct", sl))
                this_tp = float(entry_row.get("tp_pct", tp))
            else:
                this_sl = sl
                this_tp = tp

            should_exit = False
            exit_type = "SELL"

            if row.get("signal", 0) == -1:
                should_exit = True
                exit_type = "SELL"
            elif pnl <= -this_sl:
                should_exit = True
                exit_type = "SL"
            elif pnl >= this_tp:
                should_exit = True
                exit_type = "TP"

            if should_exit:
                fee_total = balance * fee * 2
                balance = balance * (1.0 + pnl) - fee_total
                trades.append(
                    {
                        "time": ts,
                        "type": exit_type,
                        "price": float(row["close"]),
                        "pnl_%": round(pnl * 100, 2),
                        "balance": balance,
                    }
                )
                if pnl > 0:
                    n_wins += 1

                position = 0
                entry_price = 0.0
                entry_row = None

                # Exit → eksponering nulstilles
                risk_state = risk_engine.apply_fill(
                    risk_state,
                    new_equity=balance,
                    new_notional_exposure=0.0,
                    new_open_positions=0,
                    now=datetime.utcnow(),
                )

        # Equity-kurven opdateres for hver bar
        equity_curve.append(balance)

    # Force exit ved slut hvis position stadig åben
    if position == 1 and entry_row is not None:
        final_price = float(df.iloc[-1]["close"])
        final_ts = df.iloc[-1].get("timestamp", len(df) - 1)
        pnl = (final_price - entry_price) / entry_price
        if use_adaptive_sl_tp:
            this_sl = float(entry_row.get("sl_pct", sl))
            this_tp = float(entry_row.get("tp_pct", tp))
        else:
            this_sl = sl
            this_tp = tp

        fee_total = balance * fee * 2
        balance = balance * (1.0 + pnl) - fee_total
        equity_curve.append(balance)
        trades.append(
            {
                "time": final_ts,
                "type": "FORCE_EXIT",
                "price": final_price,
                "pnl_%": round(pnl * 100, 2),
                "balance": balance,
            }
        )

        # Opdater RiskState for force-exit
        risk_state = risk_engine.apply_fill(
            risk_state,
            new_equity=balance,
            new_notional_exposure=0.0,
            new_open_positions=0,
            now=datetime.utcnow(),
        )

    # Gem journal
    trades_df = pd.DataFrame(trades)
    os.makedirs(JOURNAL_PATH.parent, exist_ok=True)
    trades_df.to_csv(str(JOURNAL_PATH), index=False)

    win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    print(f"Slutbalance: {balance:.2f}")
    print(f"Antal handler: {n_trades}")
    print(f"Win-rate: {win_rate:.1f}%")
    print(f"Journal gemt: {JOURNAL_PATH}")

    print_performance_report(equity_curve=equity_curve, trades_df=trades_df)
    plot_trades(df, trades_df, JOURNAL_PATH)

    return balance, trades_df


if __name__ == "__main__":
    for symbol in COINS:
        for tf in TIMEFRAMES:
            feature_path = find_latest_feature_file(symbol, tf, version="v1.3")
            if not feature_path:
                print(f"❌ Featurefil mangler: {symbol} {tf} version v1.3")
                continue

            print(f"\n=== Backtester {symbol} {tf} med Voting-Ensemble ===")
            df = pd.read_csv(feature_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Uncomment for adaptive SL/TP
            # df = add_adaptive_sl_tp(df)

            # Vælg strategi (kan byttes ud let)
            # df = ema_crossover_strategy(df)
            # df = ema_rsi_regime_strategy(df)
            # df = ema_rsi_adx_strategy(df)
            df = voting_ensemble(df)

            journal_path = (
                Path(PROJECT_ROOT)
                / "outputs"
                / f"paper_trades_{symbol.lower()}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            paper_trade(df, JOURNAL_PATH=journal_path)
