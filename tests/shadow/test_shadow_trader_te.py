# tests/shadow/test_shadow_trader_te.py

import datetime as dt

from bot.shadow.shadow_trader import ShadowTrader, TradePair


class DummyStrategy:
    def on_bar(self, bar):
        self.last_bar = bar


class DummyFillEngine:
    def __init__(self, price: float):
        self.price = price

    def simulate_fill(self, order):
        return self.price


def test_te_metrics_computed():
    st = DummyStrategy()
    fe = DummyFillEngine(price=100.0)
    shadow = ShadowTrader(st, fe, output_dir="outputs_test")

    now = dt.datetime(2025, 1, 1, 12, 0, 0)

    shadow.record_fill_pair(
        ts=now,
        symbol="BTCUSDT",
        side="BUY",
        qty=0.1,
        price_real=101.0,
        order_obj={"dummy": True},
    )

    m = shadow.aggregate_metrics()
    assert m["n_trades"] == 1
    assert abs(m["te_mean"] - 0.01) < 1e-6  # 1 % TE
