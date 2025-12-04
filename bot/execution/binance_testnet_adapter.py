# bot/execution/binance_testnet_adapter.py

"""
Binance spot TESTNET execution-adapter til paper trading.

Formål:
- Modtage interne InternalOrder-objekter fra engine.
- Kalde Binance spot testnet REST API.
- Returnere ExecutionResult med status/fills.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


# TODO: Tilpas disse imports til jeres faktiske modeller.
# Hvis I allerede har InternalOrder/ExecutionResult/OrderSide/OrderType i fx
# bot.execution.paper_adapter eller et models-modul, så importér dem derfra.
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class InternalOrder:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class Fill:
    price: float
    qty: float
    commission: float
    commission_asset: str


@dataclass
class ExecutionResult:
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: str
    executed_qty: float
    avg_price: Optional[float]
    fills: List[Fill]
    client_order_id: Optional[str]
    exchange_order_id: Optional[int]
    raw: Dict[str, Any]


@dataclass
class BinanceExecutionConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://testnet.binance.vision"
    recv_window: int = 5000
    timeout_sec: int = 10
    max_retries: int = 3
    retry_backoff_sec: float = 1.0


class BinanceTestnetExecutionAdapter:
    """
    Simpel REST-baseret execution-adapter til Binance spot TESTNET.

    Bruger signeret /api/v3/order endpoint.
    """

    def __init__(self, config: BinanceExecutionConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.config.api_key})

    # ---------- Public API ----------

    def place_order(self, order: InternalOrder) -> ExecutionResult:
        """
        Placerer en ordre på Binance testnet og returnerer ExecutionResult.

        Raises:
            RuntimeError ved ikke-håndterede API-fejl.
        """
        payload = self._build_new_order_payload(order)
        logger.info("Placing Binance testnet order", extra={"payload": payload})

        data = self._signed_request("POST", "/api/v3/order", payload)

        logger.info(
            "Binance order response",
            extra={
                "symbol": order.symbol,
                "clientOrderId": data.get("clientOrderId"),
                "status": data.get("status"),
            },
        )

        return self._parse_execution_result(order, data)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[ExecutionResult]:
        """
        Henter åbne ordrer og mapper dem til ExecutionResult (uden fills).
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()

        data = self._signed_request("GET", "/api/v3/openOrders", params)

        results: List[ExecutionResult] = []
        for entry in data:
            results.append(
                ExecutionResult(
                    symbol=entry["symbol"],
                    side=OrderSide(entry["side"]),
                    order_type=OrderType(entry["type"]),
                    status=entry["status"],
                    executed_qty=float(entry["executedQty"]),
                    avg_price=None,
                    fills=[],
                    client_order_id=entry.get("clientOrderId"),
                    exchange_order_id=entry.get("orderId"),
                    raw=entry,
                )
            )
        return results

    def cancel_order(
        self,
        symbol: str,
        client_order_id: Optional[str] = None,
        order_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Annullerer en ordre identificeret ved clientOrderId eller orderId.
        """
        if not client_order_id and not order_id:
            raise ValueError("Enten client_order_id eller order_id skal angives")

        payload: Dict[str, Any] = {"symbol": symbol.upper()}
        if client_order_id:
            payload["origClientOrderId"] = client_order_id
        if order_id:
            payload["orderId"] = order_id

        data = self._signed_request("DELETE", "/api/v3/order", payload)
        logger.info(
            "Binance cancel response",
            extra={
                "symbol": symbol,
                "clientOrderId": client_order_id,
                "orderId": order_id,
                "status": data.get("status"),
            },
        )
        return data

    # ---------- Intern helper-logik ----------

    def _build_new_order_payload(self, order: InternalOrder) -> Dict[str, Any]:
        if order.client_order_id is None:
            # Simpel clientOrderId – i praksis vil I typisk have jeres egen generator
            order.client_order_id = f"lyra_{int(time.time() * 1000)}"

        payload: Dict[str, Any] = {
            "symbol": order.symbol.upper(),
            "side": order.side.value,
            "type": order.order_type.value,
            "newClientOrderId": order.client_order_id,
        }

        # Binance kræver enten quantity eller quoteOrderQty – vi bruger quantity her
        payload["quantity"] = self._format_qty(order.quantity)

        if order.order_type == OrderType.LIMIT:
            if order.price is None:
                raise ValueError("LIMIT-ordre kræver en pris")
            payload["price"] = self._format_price(order.price)
            payload["timeInForce"] = "GTC"

        return payload

    def _parse_execution_result(
        self, order: InternalOrder, data: Dict[str, Any]
    ) -> ExecutionResult:
        fills: List[Fill] = []
        raw_fills = data.get("fills", [])
        for f in raw_fills:
            fills.append(
                Fill(
                    price=float(f["price"]),
                    qty=float(f["qty"]),
                    commission=float(f.get("commission", 0.0)),
                    commission_asset=f.get("commissionAsset", ""),
                )
            )

        executed_qty = float(data.get("executedQty", 0.0))
        # Beregn gennemsnitspris hvis muligt
        if fills:
            total_notional = sum(fl.price * fl.qty for fl in fills)
            avg_price: Optional[float] = total_notional / executed_qty if executed_qty > 0 else None
        else:
            avg_price = None

        return ExecutionResult(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            status=data["status"],
            executed_qty=executed_qty,
            avg_price=avg_price,
            fills=fills,
            client_order_id=data.get("clientOrderId"),
            exchange_order_id=data.get("orderId"),
            raw=data,
        )

    # ---------- HTTP / signatur ----------

    def _signed_request(self, method: str, path: str, params: Dict[str, Any]) -> Any:
        """
        Udfører et signeret Binance REST-kald med simpel retry/backoff.
        Logger og raiser til sidst med status + body, hvis alle forsøg fejler.

        VIGTIGT: vi signer på præcis den samme query-string (rækkefølge)
        som vi sender i HTTP-kaldet – derfor ingen sortering.
        """
        url = self.config.base_url.rstrip("/") + path
        last_resp: Optional[requests.Response] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                ts = int(time.time() * 1000)
                signed_params = dict(params)  # copy for ikke at mutere caller
                signed_params["timestamp"] = ts
                signed_params["recvWindow"] = self.config.recv_window

                # Byg query_string i samme rækkefølge som vi sender den
                query_string = urlencode(signed_params)
                signature = hmac.new(
                    self.config.api_secret.encode("utf-8"),
                    query_string.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                signed_params["signature"] = signature

                method_upper = method.upper()
                if method_upper == "GET":
                    resp = self.session.get(
                        url,
                        params=signed_params,
                        timeout=self.config.timeout_sec,
                    )
                elif method_upper == "POST":
                    resp = self.session.post(
                        url,
                        params=signed_params,
                        timeout=self.config.timeout_sec,
                    )
                elif method_upper == "DELETE":
                    resp = self.session.delete(
                        url,
                        params=signed_params,
                        timeout=self.config.timeout_sec,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                last_resp = resp

                if resp.status_code == 200:
                    return resp.json()

                logger.warning(
                    "Binance API error",
                    extra={
                        "status_code": resp.status_code,
                        "body": resp.text,
                        "attempt": attempt,
                        "path": path,
                    },
                )

                # Hårde fejl – ikke værd at retry'e
                if resp.status_code in (401, 403, 429):
                    resp.raise_for_status()

            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Binance request failed",
                    extra={"attempt": attempt, "path": path},
                )
                if attempt == self.config.max_retries:
                    raise RuntimeError(
                        f"Binance request failed after {attempt} attempts",
                    ) from exc

            # Backoff mellem forsøg
            time.sleep(self.config.retry_backoff_sec)

        if last_resp is not None:
            # Alle forsøg fejlede – giv klar fejlbesked
            raise RuntimeError(
                f"Binance request failed after {self.config.max_retries} attempts "
                f"(path={path}, status={last_resp.status_code}, body={last_resp.text})",
            )

        raise RuntimeError("Binance request failed without HTTP response")

    @staticmethod
    def _format_qty(qty: float) -> str:
        # Kan tilpasses symbol-specifik præcision senere
        return f"{qty:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_price(price: float) -> str:
        return f"{price:.6f}".rstrip("0").rstrip(".")
