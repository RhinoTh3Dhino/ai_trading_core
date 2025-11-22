# tests/backtest/test_fill_engine_v2_unit.py
import numpy as np
import pytest

from backtest.fill_engine_v2 import (
    BacktestOrder,
    FillEngineConfig,
    FillEngineV2,
    MarketSnapshot,
    OrderType,
    Side,
    TimeInForce,
)


def make_engine(max_slippage_bps: float = 0.0, impact_k: float = 0.0) -> FillEngineV2:
    cfg = FillEngineConfig(
        latency_ms=0,
        impact_k=impact_k,
        max_slippage_bps=max_slippage_bps,
    )
    return FillEngineV2(cfg)


def make_snapshot(
    bid: float = 99.0,
    ask: float = 101.0,
    bid_size: float = 10.0,
    ask_size: float = 10.0,
    ts: int = 0,
) -> MarketSnapshot:
    return MarketSnapshot(
        ts=ts,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
    )


def make_order(
    side: Side,
    qty: float = 1.0,
    ts: int = 0,
    order_type: OrderType = OrderType.MARKET,
    tif: TimeInForce = TimeInForce.IOC,
) -> BacktestOrder:
    return BacktestOrder(
        order_id="test",
        symbol="BT",
        side=side,
        order_type=order_type,
        time_in_force=tif,
        qty=qty,
        submit_ts=ts,
    )


def _avg_fill_price(res) -> float:
    assert res.fills, "Forventer mindst én fill"
    qtys = [f.qty for f in res.fills]
    prices = [f.price for f in res.fills]
    return float(np.average(prices, weights=qtys))


def test_buy_market_ioc_full_fill():
    engine = make_engine()
    snapshot = make_snapshot(bid=99.0, ask=101.0, bid_size=10.0, ask_size=10.0)
    order = make_order(side=Side.BUY, qty=1.0)

    res = engine.simulate_order(order, snapshot)

    # Kontrakt-tests
    assert res.status in {"FILLED", "PARTIALLY_FILLED"}
    assert res.fills
    total_qty = sum(f.qty for f in res.fills)
    assert pytest.approx(total_qty, rel=1e-6) == 1.0
    px = _avg_fill_price(res)
    # Mindst inden for bid/ask-spread
    assert snapshot.bid <= px <= snapshot.ask


def test_sell_market_ioc_full_fill():
    engine = make_engine()
    snapshot = make_snapshot(bid=99.0, ask=101.0, bid_size=10.0, ask_size=10.0)
    order = make_order(side=Side.SELL, qty=2.0)

    res = engine.simulate_order(order, snapshot)

    assert res.status in {"FILLED", "PARTIALLY_FILLED"}
    total_qty = sum(f.qty for f in res.fills)
    assert pytest.approx(total_qty, rel=1e-6) == 2.0
    px = _avg_fill_price(res)
    assert snapshot.bid <= px <= snapshot.ask


def test_partial_fill_begrænset_af_liquidity():
    engine = make_engine()
    snapshot = make_snapshot(bid=99.0, ask=101.0, bid_size=0.5, ask_size=0.5)
    order = make_order(side=Side.BUY, qty=2.0)  # mere end ask_size

    res = engine.simulate_order(order, snapshot)

    # Vi forventer max ask_size fyldes (eller mindre) – ikke mere end likviditeten
    total_qty = sum(f.qty for f in res.fills)
    assert total_qty <= snapshot.ask_size + 1e-9
    assert res.status in {"PARTIALLY_FILLED", "CANCELLED", "OPEN"}


def test_reject_nar_der_ingen_likviditet():
    engine = make_engine()
    snapshot = make_snapshot(bid=0.0, ask=0.0, bid_size=0.0, ask_size=0.0)
    order = make_order(side=Side.BUY, qty=1.0)

    res = engine.simulate_order(order, snapshot)

    # Ingen fills, status = CANCELLED i vores implementation
    assert not res.fills
    assert res.status in {"CANCELLED", "REJECTED"}


def test_slippage_respekterer_max_slippage_bps():
    max_slip = 50  # 0,50 %
    engine = make_engine(max_slippage_bps=max_slip, impact_k=1.0)
    snapshot = make_snapshot(bid=100.0, ask=101.0, bid_size=10.0, ask_size=10.0)
    order = make_order(side=Side.BUY, qty=5.0)

    res = engine.simulate_order(order, snapshot)
    assert res.fills

    px = _avg_fill_price(res)
    max_allowed = snapshot.ask * (1.0 + max_slip / 1e4)
    # Pris må ikke flytte sig mere end max_slippage_bps
    assert px <= max_allowed + 1e-9
