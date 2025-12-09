# bot/execution/paper_adapter.py

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class VenueClient(Protocol):
    """Minimal interface som dine venue-klienter skal opfylde."""

    def place_order(
        self, *, symbol: str, side: str, qty: float, price: float, client_order_id: str
    ) -> dict: ...

    def cancel_order(self, *, symbol: str, client_order_id: str) -> dict: ...

    def get_open_orders(self, *, symbol: Optional[str] = None) -> list[dict]: ...


@dataclass
class ExecutionConfig:
    venue: str
    mode: str  # "paper" | "testnet"
    max_retries: int = 3
    backoff_seconds: float = 0.5
    dead_letter_path: str = "logs/deadletter_orders.jsonl"
    max_inflight_orders: int = 100


@dataclass
class OrderRequest:
    strategy_id: str
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    price: float
    ts_ns: int  # strategi-timestamp (nanosekunder)


@dataclass
class OrderResult:
    ok: bool
    client_order_id: str
    venue_order_id: Optional[str]
    error: Optional[str] = None
    raw: Optional[dict] = None


class PaperExecutionAdapter:
    """
    Wrapper omkring VenueClient der håndterer:
    - Idempotent clientOrderId
    - Retries + backoff
    - Dead-letter ved permanente fejl
    - Simpel måling af ack-latens (for p95/p99 metrics)
    """

    def __init__(self, venue_client: VenueClient, config: ExecutionConfig) -> None:
        self._client = venue_client
        self._cfg = config

    # ---------- Public API ----------

    def submit_order(self, req: OrderRequest) -> OrderResult:
        client_order_id = self._make_client_order_id(req)
        start = time.monotonic()

        last_error: Optional[str] = None
        response: Optional[dict] = None

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                response = self._client.place_order(
                    symbol=req.symbol,
                    side=req.side,
                    qty=req.qty,
                    price=req.price,
                    client_order_id=client_order_id,
                )
                latency_ms = (time.monotonic() - start) * 1000
                logger.info(
                    "paper_order_ok",
                    extra={
                        "client_order_id": client_order_id,
                        "latency_ms": latency_ms,
                        "attempt": attempt,
                    },
                )
                return OrderResult(
                    ok=True,
                    client_order_id=client_order_id,
                    venue_order_id=response.get("order_id") if response else None,
                    raw=response,
                )
            except Exception as exc:  # pragma: no cover - konkretiseres senere
                last_error = str(exc)
                logger.warning(
                    "paper_order_retry",
                    extra={
                        "client_order_id": client_order_id,
                        "attempt": attempt,
                        "error": last_error,
                    },
                )
                if attempt >= self._cfg.max_retries:
                    break
                time.sleep(self._cfg.backoff_seconds * attempt)

        # Dead-letter
        self._write_dead_letter(req, client_order_id, last_error)
        return OrderResult(
            ok=False,
            client_order_id=client_order_id,
            venue_order_id=None,
            error=last_error,
        )

    def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        try:
            self._client.cancel_order(symbol=symbol, client_order_id=client_order_id)
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(
                "paper_cancel_failed",
                extra={"client_order_id": client_order_id, "error": str(exc)},
            )
            return False

    # ---------- Intern hjælpe-logik ----------

    def _make_client_order_id(self, req: OrderRequest) -> str:
        """
        Idempotent clientOrderId baseret på deterministisk hash af ordredata.
        Sørg for at req.ts_ns er stabil ved retries.
        """
        payload = (
            f"{req.strategy_id}|{req.symbol}|{req.side}|"
            f"{req.qty:.8f}|{req.price:.8f}|{req.ts_ns}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{self._cfg.mode.upper()}_{digest}"

    def _write_dead_letter(self, req: OrderRequest, cid: str, error: Optional[str]) -> None:
        import json
        from pathlib import Path

        Path(self._cfg.dead_letter_path).parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "client_order_id": cid,
            "strategy_id": req.strategy_id,
            "symbol": req.symbol,
            "side": req.side,
            "qty": req.qty,
            "price": req.price,
            "ts_ns": req.ts_ns,
            "error": error,
            "venue": self._cfg.venue,
            "mode": self._cfg.mode,
            "created_ts": time.time(),
        }
        with open(self._cfg.dead_letter_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
