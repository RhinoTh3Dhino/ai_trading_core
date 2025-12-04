# tests/risk/test_risk_engine.py

import datetime as dt

from risk.risk_engine import RiskEngine, RiskLimits, RiskState


def _default_limits() -> RiskLimits:
    return RiskLimits(
        max_daily_loss_pct=0.03,
        max_equity_drawdown_pct=0.10,
        max_notional_exposure_pct=0.50,
        max_per_trade_risk_pct=0.01,
        max_open_positions=5,
        circuit_breaker_drawdown_pct=0.05,
        circuit_breaker_cooldown_minutes=60,
    )


def test_pre_trade_ok_under_limits():
    limits = _default_limits()
    engine = RiskEngine(limits)

    state = RiskState(
        trading_day=dt.date.today(),
        start_equity=100_000,
        high_equity=100_000,
        current_equity=100_000,
        notional_exposure=10_000,
        open_positions=2,
    )

    decision = engine.check_pre_trade(state, order_notional=1_000)
    assert decision.allowed


def test_block_when_daily_loss_exceeded():
    limits = _default_limits()
    engine = RiskEngine(limits)

    state = RiskState(
        trading_day=dt.date.today(),
        start_equity=100_000,
        high_equity=100_000,
        current_equity=96_000,  # -4 %
        notional_exposure=0,
        open_positions=0,
    )

    decision = engine.check_pre_trade(state, order_notional=1_000)
    assert not decision.allowed
    assert decision.limit_name == "max_daily_loss_pct"
