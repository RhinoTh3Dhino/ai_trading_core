# bot/brokers/ccxt_broker.py
from __future__ import annotations

import math
from typing import Optional

from data.live_feed import make_exchange


def market_qty_for_quote(
    price: float, quote_amount: float, step: float = 0.0001
) -> float:
    if price <= 0:
        return 0.0
    qty = quote_amount / price
    # rund ned til step
    return math.floor(qty / step) * step


class CcxtBroker:
    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str],
        secret: Optional[str],
        live: bool,
    ):
        self.live = live
        self.ex = make_exchange(
            exchange_id, api_key if live else None, secret if live else None
        )

    def market_buy(self, symbol: str, quote_amount: float):
        ticker = self.ex.fetch_ticker(symbol)
        price = float(ticker["last"])
        qty = market_qty_for_quote(price, quote_amount)
        if not self.live:
            return {"ok": True, "pseudo_order": f"BUY {symbol} {qty} @ ~{price}"}
        return self.ex.create_order(symbol, "market", "buy", qty)

    def market_sell(self, symbol: str, base_qty: float):
        if not self.live:
            return {"ok": True, "pseudo_order": f"SELL {symbol} {base_qty} @ market"}
        return self.ex.create_order(symbol, "market", "sell", base_qty)
