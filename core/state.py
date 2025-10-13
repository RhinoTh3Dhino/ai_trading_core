# core/state.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class PosSide(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def side_from_qty(qty: float) -> PosSide:
    if qty > 0:
        return PosSide.LONG
    if qty < 0:
        return PosSide.SHORT
    return PosSide.FLAT


@dataclass
class PosState:
    """
    Letvægts positionstilstand for et symbol.
    - side:   FLAT/LONG/SHORT
    - qty:    netto-antal (positiv = long, negativ = short)
    - avg_price: VWAP for den åbne netto-position
    - last_update_ts: UTC-tidspunkt for seneste opdatering
    """

    symbol: str
    side: PosSide  # FLAT/LONG/SHORT
    qty: float
    avg_price: float
    last_update_ts: datetime

    def __post_init__(self) -> None:
        # Sørg for, at timestamp er sat og er UTC-aware
        if self.last_update_ts is None:
            self.last_update_ts = _now_utc()
        elif self.last_update_ts.tzinfo is None:
            # antag UTC hvis naiv
            self.last_update_ts = self.last_update_ts.replace(tzinfo=timezone.utc)
        else:
            self.last_update_ts = self.last_update_ts.astimezone(timezone.utc)

    @classmethod
    def empty(cls, symbol: str, ts: Optional[datetime] = None) -> "PosState":
        """Hjælper til at skabe en FLAT position."""
        return cls(
            symbol=symbol,
            side=PosSide.FLAT,
            qty=0.0,
            avg_price=0.0,
            last_update_ts=ts or _now_utc(),
        )

    @classmethod
    def from_qty(
        cls, symbol: str, qty: float, avg_price: float, ts: Optional[datetime] = None
    ) -> "PosState":
        """Sæt side automatisk ud fra qty."""
        return cls(
            symbol=symbol,
            side=side_from_qty(qty),
            qty=qty,
            avg_price=avg_price,
            last_update_ts=ts or _now_utc(),
        )

    def mark_updated(self, ts: Optional[datetime] = None) -> None:
        """Opdater blot timestamp (UTC)."""
        self.last_update_ts = ts or _now_utc()
        if self.last_update_ts.tzinfo is None:
            self.last_update_ts = self.last_update_ts.replace(tzinfo=timezone.utc)
        else:
            self.last_update_ts = self.last_update_ts.astimezone(timezone.utc)

    def set_flat(self, ts: Optional[datetime] = None) -> None:
        """Nulstil til FLAT uden position."""
        self.qty = 0.0
        self.avg_price = 0.0
        self.side = PosSide.FLAT
        self.mark_updated(ts)

    def net_value(self, price: float) -> float:
        """Brugbar helper: markedsværdi (qty * pris)."""
        return float(self.qty) * float(price)
