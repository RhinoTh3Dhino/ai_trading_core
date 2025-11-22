# tests/backtest/test_fill_engine_v2.py

from backtest.fill_engine_v2 import (
    BacktestOrder,
    FillEngineConfig,
    FillEngineV2,
    MarketSnapshot,
    OrderType,
    Side,
    TimeInForce,
)


def _default_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        ts=1_700_000_000_000,
        bid=99.0,
        ask=101.0,
        bid_size=5.0,
        ask_size=5.0,
    )


def test_market_buy_fills_at_ask_with_latency():
    engine = FillEngineV2(FillEngineConfig(latency_ms=50, impact_k=0.0))
    order = BacktestOrder(
        order_id="o1",
        symbol="TESTUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
        qty=1.0,
    )
    snap = _default_snapshot()

    res = engine.simulate_order(order, snap)

    assert res.status == "FILLED"
    assert res.remaining_qty == 0.0
    assert len(res.fills) == 1
    fill = res.fills[0]
    assert fill.price == snap.ask
    assert fill.ts == snap.ts + engine.config.latency_ms


def test_limit_buy_not_filled_if_price_above_limit_ioc():
    engine = FillEngineV2()
    order = BacktestOrder(
        order_id="o2",
        symbol="TESTUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.IOC,
        qty=1.0,
        limit_price=100.0,
    )
    snap = _default_snapshot()  # ask=101

    res = engine.simulate_order(order, snap)

    assert res.status == "CANCELLED"
    assert res.remaining_qty == 1.0
    assert res.fills == []


def test_limit_buy_partial_fill_ioc():
    engine = FillEngineV2()
    order = BacktestOrder(
        order_id="o3",
        symbol="TESTUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.IOC,
        qty=10.0,
        limit_price=101.0,
    )
    snap = _default_snapshot()  # ask_size=5 < qty=10

    res = engine.simulate_order(order, snap)

    assert res.status == "PARTIALLY_FILLED"
    assert res.remaining_qty == 5.0
    assert len(res.fills) == 1
    assert res.fills[0].qty == 5.0
    assert res.fills[0].is_partial is True


def test_limit_buy_fok_requires_full_qty():
    engine = FillEngineV2()
    order = BacktestOrder(
        order_id="o4",
        symbol="TESTUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.FOK,
        qty=10.0,
        limit_price=101.0,
    )
    snap = _default_snapshot()  # ask_size=5 < qty=10

    res = engine.simulate_order(order, snap)

    assert res.status == "CANCELLED"
    assert res.remaining_qty == 10.0
    assert res.fills == []


def test_impact_model_moves_price_and_caps_slippage():
    cfg = FillEngineConfig(latency_ms=0, impact_k=0.5, max_slippage_bps=50.0)
    engine = FillEngineV2(cfg)

    order = BacktestOrder(
        order_id="o5",
        symbol="TESTUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
        qty=100.0,
    )
    snap = _default_snapshot()

    res = engine.simulate_order(order, snap)
    fill = res.fills[0]

    assert fill.price >= snap.ask
    max_move = snap.ask * cfg.max_slippage_bps / 1e4
    assert fill.price <= snap.ask + max_move + 1e-9
