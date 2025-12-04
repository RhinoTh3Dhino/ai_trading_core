# tests/execution/test_binance_testnet_adapter.py

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.execution.binance_testnet_adapter import (
    BinanceExecutionConfig,
    BinanceTestnetExecutionAdapter,
    InternalOrder,
    OrderSide,
    OrderType,
)


class DummyResponse:
    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data
        self.text = str(json_data)

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:  # pragma: no cover
        raise RuntimeError(f"HTTP {self.status_code}")


def test_place_order_market_success(monkeypatch: pytest.MonkeyPatch) -> None:
    config = BinanceExecutionConfig(api_key="x", api_secret="y")
    adapter = BinanceTestnetExecutionAdapter(config)

    # Mock session.post
    def fake_post(url, params, timeout):  # noqa: ANN001
        data = {
            "symbol": params["symbol"],
            "side": params["side"],
            "type": params["type"],
            "status": "FILLED",
            "executedQty": "1.0",
            "clientOrderId": params.get("newClientOrderId", "cid"),
            "orderId": 123456,
            "fills": [
                {"price": "100.0", "qty": "1.0", "commission": "0.0", "commissionAsset": "USDT"},
            ],
        }
        return DummyResponse(200, data)

    adapter.session = SimpleNamespace(post=fake_post)  # type: ignore[assignment]

    order = InternalOrder(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )

    result = adapter.place_order(order)

    assert result.status == "FILLED"
    assert result.executed_qty == 1.0
    assert result.avg_price == pytest.approx(100.0)
    assert result.symbol == "BTCUSDT"
