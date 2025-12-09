# risk/risk_engine.py

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskLimits:
    max_daily_loss_pct: float
    max_equity_drawdown_pct: float
    max_notional_exposure_pct: float
    max_per_trade_risk_pct: float
    max_open_positions: int
    circuit_breaker_drawdown_pct: float
    circuit_breaker_cooldown_minutes: int


@dataclass
class RiskState:
    trading_day: dt.date
    start_equity: float
    high_equity: float
    current_equity: float
    notional_exposure: float
    open_positions: int
    circuit_breaker_triggered_at: Optional[dt.datetime] = None


@dataclass
class RiskDecision:
    allowed: bool
    reason: Optional[str] = None
    limit_name: Optional[str] = None


class RiskEngine:
    """
    Stateless logik + enkel RiskState der kan ligge i memory eller persisteres.

    Brug:
    - kald check_pre_trade() før du sender ordre til execution adapter.
    - kald apply_fill() efter fills for at opdatere equity/exposure.
    """

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    # ---------- Public API ----------

    def check_pre_trade(self, state: RiskState, *, order_notional: float) -> RiskDecision:
        # Circuit breaker aktiv?
        if self._circuit_breaker_active(state):
            return RiskDecision(
                allowed=False,
                limit_name="circuit_breaker",
                reason="Circuit breaker aktiv – cooldown periode ikke udløbet",
            )

        # Dagligt tab
        daily_pnl_pct = (state.current_equity - state.start_equity) / state.start_equity
        if daily_pnl_pct <= -self._limits.max_daily_loss_pct:
            return RiskDecision(
                allowed=False,
                limit_name="max_daily_loss_pct",
                reason=f"Dagligt tab {daily_pnl_pct:.4f} over limit",
            )

        # Drawdown fra intradag high
        dd_pct = (state.current_equity - state.high_equity) / state.high_equity
        if dd_pct <= -self._limits.max_equity_drawdown_pct:
            return RiskDecision(
                allowed=False,
                limit_name="max_equity_drawdown_pct",
                reason=f"Drawdown {dd_pct:.4f} over limit",
            )

        # Per trade risk
        per_trade_risk_pct = order_notional / state.current_equity
        if per_trade_risk_pct > self._limits.max_per_trade_risk_pct:
            return RiskDecision(
                allowed=False,
                limit_name="max_per_trade_risk_pct",
                reason=f"Order notional {per_trade_risk_pct:.4f} over limit",
            )

        # Samlet eksponering
        total_exposure = state.notional_exposure + order_notional
        total_exposure_pct = total_exposure / state.current_equity
        if total_exposure_pct > self._limits.max_notional_exposure_pct:
            return RiskDecision(
                allowed=False,
                limit_name="max_notional_exposure_pct",
                reason=f"Exposure {total_exposure_pct:.4f} over limit",
            )

        # Antal åbne positioner
        if state.open_positions >= self._limits.max_open_positions:
            return RiskDecision(
                allowed=False,
                limit_name="max_open_positions",
                reason="For mange åbne positioner",
            )

        return RiskDecision(allowed=True)

    def apply_fill(
        self,
        state: RiskState,
        *,
        new_equity: float,
        new_notional_exposure: float,
        new_open_positions: int,
        now: Optional[dt.datetime] = None,
    ) -> RiskState:
        now = now or dt.datetime.utcnow()

        high_equity = max(state.high_equity, new_equity)
        new_state = RiskState(
            trading_day=state.trading_day,
            start_equity=state.start_equity,
            high_equity=high_equity,
            current_equity=new_equity,
            notional_exposure=new_notional_exposure,
            open_positions=new_open_positions,
            circuit_breaker_triggered_at=state.circuit_breaker_triggered_at,
        )

        # Sæt circuit breaker hvis drawdown overstiger limit
        dd_pct = (new_equity - high_equity) / high_equity
        if (
            dd_pct <= -self._limits.circuit_breaker_drawdown_pct
            and new_state.circuit_breaker_triggered_at is None
        ):
            new_state.circuit_breaker_triggered_at = now

        return new_state

    # ---------- Internt ----------

    def _circuit_breaker_active(self, state: RiskState) -> bool:
        if state.circuit_breaker_triggered_at is None:
            return False
        delta = dt.datetime.utcnow() - state.circuit_breaker_triggered_at
        return delta.total_seconds() < self._limits.circuit_breaker_cooldown_minutes * 60
