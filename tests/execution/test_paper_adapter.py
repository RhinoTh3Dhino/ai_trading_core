# tests/execution/test_paper_adapter.py

from bot.execution.paper_adapter import (
    ExecutionConfig,
    OrderRequest,
    PaperExecutionAdapter,
)


class DummyOkVenueClient:
    def __init__(self):
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        return {"order_id": "TEST123", "echo": kwargs}

    def cancel_order(self, **kwargs):
        return {"status": "OK"}


def test_client_order_id_is_deterministic():
    cfg = ExecutionConfig(venue="dummy", mode="paper")
    adapter = PaperExecutionAdapter(DummyOkVenueClient(), cfg)

    req = OrderRequest(
        strategy_id="s1",
        symbol="BTCUSDT",
        side="BUY",
        qty=0.1,
        price=50000.0,
        ts_ns=123456789,
    )

    cid1 = adapter._make_client_order_id(req)
    cid2 = adapter._make_client_order_id(req)

    assert cid1 == cid2
    assert cid1.startswith("PAPER_")


def test_submit_order_success():
    cfg = ExecutionConfig(venue="dummy", mode="paper", max_retries=2)
    dummy = DummyOkVenueClient()
    adapter = PaperExecutionAdapter(dummy, cfg)

    req = OrderRequest(
        strategy_id="s1",
        symbol="BTCUSDT",
        side="BUY",
        qty=0.1,
        price=50000.0,
        ts_ns=123456789,
    )
    result = adapter.submit_order(req)

    assert result.ok
    assert result.venue_order_id == "TEST123"
    assert dummy.calls == 1
