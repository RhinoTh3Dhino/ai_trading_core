# backtest/fill_engine_v2.py
"""
Fill-motor v2 for mere realistisk backtest.

Funktionalitet:
- Partial fills
- LIMIT / MARKET / IOC / FOK
- Venue-latency (ms) og slippage
- Simpel impact: price += impact_k * sqrt(executed_qty)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import List, Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good-till-cancel
    IOC = "IOC"  # Immediate-or-cancel (partial ok)
    FOK = "FOK"  # Fill-or-kill (kun fuld fill, ellers 0)


@dataclass
class BacktestOrder:
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    time_in_force: TimeInForce
    qty: float
    limit_price: Optional[float] = None
    submit_ts: int = 0  # ms since epoch


@dataclass
class MarketSnapshot:
    """
    Forenklet market snapshot; kan mappes til dine eksisterende bar/book-typer.
    """

    ts: int  # ms
    bid: float
    ask: float
    bid_size: float
    ask_size: float


@dataclass
class Fill:
    order_id: str
    symbol: str
    ts: int
    side: Side
    qty: float
    price: float
    is_partial: bool
    venue: str = "sim"


@dataclass
class FillResult:
    order: BacktestOrder
    fills: List[Fill]
    remaining_qty: float
    status: str  # "FILLED", "PARTIALLY_FILLED", "CANCELLED", "OPEN", "REJECTED"


@dataclass
class FillEngineConfig:
    latency_ms: int = 50  # simuleret venue-latency
    impact_k: float = 0.0  # 0.0 = ingen impact
    max_slippage_bps: float = 5.0  # cap på slippage ift. base price


class FillEngineV2:
    """
    Fill-engine der køres i backtest-loopet.

    Forventet brug:
        engine = FillEngineV2(config)
        result = engine.simulate_order(order, snapshot)
    """

    def __init__(self, config: Optional[FillEngineConfig] = None):
        self.config = config or FillEngineConfig()

    # ---------- Public API ----------

    def simulate_order(self, order: BacktestOrder, snapshot: MarketSnapshot) -> FillResult:
        if order.qty <= 0:
            return FillResult(order, [], 0.0, "REJECTED")

        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, snapshot)

        if order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, snapshot)

        return FillResult(order, [], order.qty, "REJECTED")

    # ---------- Intern logik ----------

    def _fill_market(self, order: BacktestOrder, snapshot: MarketSnapshot) -> FillResult:
        exec_ts = snapshot.ts + self.config.latency_ms

        if order.side == Side.BUY:
            base_price = snapshot.ask
            available = snapshot.ask_size
        else:
            base_price = snapshot.bid
            available = snapshot.bid_size

        if available <= 0:
            return FillResult(order, [], order.qty, "CANCELLED")

        exec_qty = min(order.qty, available)
        price = self._apply_impact_and_slippage(base_price, exec_qty, order.side)

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            ts=exec_ts,
            side=order.side,
            qty=exec_qty,
            price=price,
            is_partial=exec_qty < order.qty,
        )

        remaining = order.qty - exec_qty
        status = "FILLED" if remaining == 0 else "PARTIALLY_FILLED"

        return FillResult(order, [fill], remaining, status)

    def _fill_limit(self, order: BacktestOrder, snapshot: MarketSnapshot) -> FillResult:
        assert order.limit_price is not None, "LIMIT ordre kræver limit_price"
        exec_ts = snapshot.ts + self.config.latency_ms

        if order.side == Side.BUY:
            # BUY LIMIT: pris <= limit
            if snapshot.ask > order.limit_price:
                return self._handle_unfilled_limit(order)
            base_price = min(snapshot.ask, order.limit_price)
            available = snapshot.ask_size
        else:
            # SELL LIMIT: pris >= limit
            if snapshot.bid < order.limit_price:
                return self._handle_unfilled_limit(order)
            base_price = max(snapshot.bid, order.limit_price)
            available = snapshot.bid_size

        if available <= 0:
            return self._handle_unfilled_limit(order)

        exec_qty = min(order.qty, available)

        # FOK: kræver fuld qty
        if order.time_in_force == TimeInForce.FOK and exec_qty < order.qty:
            return FillResult(order, [], order.qty, "CANCELLED")

        price = self._apply_impact_and_slippage(base_price, exec_qty, order.side)

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            ts=exec_ts,
            side=order.side,
            qty=exec_qty,
            price=price,
            is_partial=exec_qty < order.qty,
        )

        remaining = order.qty - exec_qty

        if remaining > 0 and order.time_in_force in (TimeInForce.IOC,):
            status = "PARTIALLY_FILLED"
        elif remaining > 0:
            status = "PARTIALLY_FILLED"  # GTC: i denne model kun ét snapshot ad gangen
        else:
            status = "FILLED"

        return FillResult(order, [fill], remaining, status)

    def _handle_unfilled_limit(self, order: BacktestOrder) -> FillResult:
        if order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
            return FillResult(order, [], order.qty, "CANCELLED")
        return FillResult(order, [], order.qty, "OPEN")

    def _apply_impact_and_slippage(self, base_price: float, qty: float, side: Side) -> float:
        if qty <= 0:
            return base_price

        impact = self.config.impact_k * sqrt(qty)
        price = base_price + impact if side == Side.BUY else base_price - impact

        max_move_abs = base_price * self.config.max_slippage_bps / 1e4
        diff = price - base_price
        if diff > max_move_abs:
            price = base_price + max_move_abs
        elif diff < -max_move_abs:
            price = base_price - max_move_abs

        return price
