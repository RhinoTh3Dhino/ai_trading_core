# bot/brokers/paper_broker.py
from __future__ import annotations

import csv
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytz


# =========================
# Datatyper / strukturer
# =========================

@dataclass
class Position:
    """Netto-position pr. symbol (qty > 0 = long, qty < 0 = short)."""
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0  # VWAP for den åbne netto-position


@dataclass
class Order:
    """Simpel ordremodel til paper trading."""
    id: str
    ts: str
    symbol: str
    side: str            # "BUY" | "SELL"
    qty: float
    type: str           # "market" | "limit"
    limit_price: Optional[float] = None
    status: str = "open"     # "open" | "filled" | "rejected" | "cancelled"
    reason: Optional[str] = None


@dataclass
class Fill:
    """Udførte handler (fills) med provision og realiseret PnL på lukkede mængder."""
    ts: str
    order_id: str
    symbol: str
    side: str           # "BUY" | "SELL"
    qty: float
    price: float
    commission: float
    pnl_realized: float  # realiseret PnL på lukkede mængder (kan være 0 ved ren åbning/tilføjelse)


# =========================
# PaperBroker
# =========================

class PaperBroker:
    """
    Enkel, robust paper broker med:
      - Market/Limit-ordrer (GTC)
      - Slippage (bp) + commission (bp)
      - Kontantkonto, netto-position pr. symbol
      - Realiseret/urealiseret PnL, equity, drawdown
      - Daglig loss limit (pct) med auto-stop af ny handel
      - CSV-logs: fills.csv og equity.csv

    VIGTIGT: Alle fills timestemples med **barens timestamp** (ts-argumentet),
    så dagsaggregering matcher korrekt mod signaler og equity.
    """

    # ------- initialisering -------

    def __init__(
        self,
        *,
        starting_cash: float = 100_000.0,
        commission_bp: float = 2.0,    # 0.02%
        slippage_bp: float = 1.0,      # 0.01%
        tz: str = "Europe/Copenhagen",
        equity_log_path: str | Path = "logs/equity.csv",
        fills_log_path: str | Path = "logs/fills.csv",
        daily_loss_limit_pct: float = 0.0,  # 0 = slukket
        allow_short: bool = False,
        price_decimals: int = 2,
        qty_decimals: int = 8,
        # støjfilter (valgfrit)
        min_qty: float = 0.0,
        min_notional: float = 0.0,
        # Default False: LIMIT-ordrer under min_notional afvises ikke ved submit
        reject_below_min: bool = False,
    ) -> None:
        self.tz = pytz.timezone(tz)
        self.price_decimals = price_decimals
        self.qty_decimals = qty_decimals
        self.min_qty = float(min_qty)
        self.min_notional = float(min_notional)
        self.reject_below_min = bool(reject_below_min)

        self.cash: float = float(starting_cash)
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.realized_pnl: float = 0.0

        self.commission_bp = float(commission_bp)
        self.slippage_bp = float(slippage_bp)
        self.allow_short = bool(allow_short)

        self.equity_log_path = Path(equity_log_path)
        self.fills_log_path = Path(fills_log_path)
        self.equity_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fills_log_path.parent.mkdir(parents=True, exist_ok=True)

        # senest kendte priser (instans-lokalt)
        self._last_prices: Dict[str, float] = {}

        self.peak_equity: float = starting_cash
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        now = self._now()
        self.daily_anchor_date = now.date()
        self.daily_start_equity: float = starting_cash
        self.trading_halted: bool = False  # sættes ved breach

        # Sørg for CSV-headere findes
        self._ensure_csv_headers()

    # ------- I/O utils -------

    def _ensure_csv_headers(self) -> None:
        if not self.fills_log_path.exists():
            with self.fills_log_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "symbol", "side", "qty", "price", "commission", "pnl_realized"])
        if not self.equity_log_path.exists():
            with self.equity_log_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "equity", "cash", "positions_value", "drawdown_pct"])

    def _append_fill(self, fill: Fill) -> None:
        with self.fills_log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([fill.ts, fill.symbol, fill.side, self._r(fill.qty, self.qty_decimals),
                        self._r(fill.price, self.price_decimals), self._r(fill.commission, 8),
                        self._r(fill.pnl_realized, 8)])

    def _append_equity_snapshot(self, date_str: str, equity: float, cash: float, pos_val: float, dd_pct: float) -> None:
        with self.equity_log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([date_str, self._r(equity, 2), self._r(cash, 2), self._r(pos_val, 2), self._r(dd_pct, 2)])

    # ------- tids- & afrundingshjælpere -------

    def _now(self) -> datetime:
        """Lokaltidsstempel i valgt tidszone."""
        return datetime.now(timezone.utc).astimezone(self.tz)

    @staticmethod
    def _r(x: float, n: int) -> float:
        """Rund for pæn log-output (påvirker ikke interne beregninger)."""
        q = 10 ** n
        return math.floor(x * q + 0.5) / q

    # ------- offentlige API-metoder -------

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        *,
        limit_price: Optional[float] = None,
        ts: Optional[datetime] = None,
    ) -> Order:
        """
        Opret en ordre. Market fyldes straks til simuleret pris; limit lægges i orderbogen (GTC).
        Returnerer ordreobjekt (med status 'filled' eller 'open'/'rejected').

        Min-filters:
          - min_qty tjekkes altid ved submit (market+limit).
          - min_notional:
              * MARKET → tjek ved submit (mod last) og igen ved fill (mod rå pris uden slippage).
              * LIMIT  → afvis ved submit kun hvis reject_below_min=True; ellers tjek kun ved fill.
        """
        if self.trading_halted:
            return Order(
                id=str(uuid.uuid4()),
                ts=self._fmt_ts(ts),
                symbol=symbol,
                side=side.upper(),
                qty=qty,
                type=order_type.lower(),
                limit_price=limit_price,
                status="rejected",
                reason="Trading halted by daily loss limit",
            )

        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side skal være 'BUY' eller 'SELL'")

        # Støjfilter på qty ved submit
        if self.min_qty > 0.0 and abs(qty) + 1e-12 < self.min_qty:
            return Order(
                id=str(uuid.uuid4()),
                ts=self._fmt_ts(ts),
                symbol=symbol,
                side=side,
                qty=float(qty),
                type=order_type.lower(),
                limit_price=float(limit_price) if limit_price is not None else None,
                status="rejected",
                reason=f"Qty below min_qty ({self.min_qty})",
            )

        # Forbud mod short ved submit (SELL må ikke overstige eksisterende long)
        if not self.allow_short and side == "SELL":
            pos = self.positions.get(symbol)
            long_qty = max(0.0, (pos.qty if pos else 0.0))
            if long_qty <= 1e-12 or float(qty) - long_qty > 1e-12:
                return Order(
                    id=str(uuid.uuid4()),
                    ts=self._fmt_ts(ts),
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    type=order_type.lower(),
                    limit_price=float(limit_price) if limit_price is not None else None,
                    status="rejected",
                    reason="Shorting ikke tilladt (mangler long-qty til at sælge)",
                )

        order = Order(
            id=str(uuid.uuid4()),
            ts=self._fmt_ts(ts),
            symbol=symbol,
            side=side,
            qty=float(qty),
            type=order_type.lower(),
            limit_price=float(limit_price) if limit_price is not None else None,
            status="open",
        )

        last_px = self._get_last_price(symbol)

        if order.type == "market":
            price = last_px
            if price is None:
                order.status = "rejected"
                order.reason = "Ukendt markedspris (kald mark_to_market først)"
                return order

            # min_notional for MARKET ved submit (mod last)
            if self.min_notional > 0.0 and abs(order.qty) * float(price) + 1e-12 < self.min_notional:
                order.status = "rejected"
                order.reason = f"Notional below min_notional ({self.min_notional})"
                return order

            self._execute_fill(order, float(price), ts=ts)  # brug barens ts
            if order.status != "rejected":
                order.status = "filled"

        elif order.type == "limit":
            if order.limit_price is None:
                order.status = "rejected"
                order.reason = "limit_price mangler"
                return order

            # LIMIT: ved submit — afvis kun under min_notional hvis flag er True
            if self.min_notional > 0.0 and abs(order.qty) * float(order.limit_price) + 1e-12 < self.min_notional:
                if self.reject_below_min:
                    order.status = "rejected"
                    order.reason = f"Notional below min_notional ({self.min_notional})"
                    return order
                # ellers: lad ordren ligge åben

            # Kan den fyldes straks?
            if last_px is not None:
                should_fill = (side == "BUY" and last_px <= order.limit_price) or (side == "SELL" and last_px >= order.limit_price)
                if should_fill:
                    self._try_fill_limit_now(order, last_px, ts)
                    if order.status == "filled":
                        return order

            # Ellers i orderbog
            self.open_orders.append(order)

        else:
            order.status = "rejected"
            order.reason = "Ukendt ordretype"

        return order

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Annullér alle åbne ordrer (evt. kun for et symbol). Returnerer antal annullerede."""
        kept: List[Order] = []
        cancelled = 0
        for o in self.open_orders:
            if symbol is None or o.symbol == symbol:
                o.status = "cancelled"
                cancelled += 1
            else:
                kept.append(o)
        self.open_orders = kept
        return cancelled

    def close_position(self, symbol: str, ts: Optional[datetime] = None) -> Optional[Order]:
        """Luk hele netto-positionen i markedet (market ordre). Returnerer ordren, eller None hvis ingen position."""
        pos = self.positions.get(symbol)
        if not pos or abs(pos.qty) < 1e-12:
            return None
        side = "SELL" if pos.qty > 0 else "BUY"
        return self.submit_order(symbol, side, abs(pos.qty), "market", ts=ts)

    def mark_to_market(self, prices: Dict[str, float], ts: Optional[datetime] = None) -> Dict:
        """
        Opdater markedspriser og forsøg at fylde limit-ordrer, beregn equity/drawdown,
        og skriv et snapshot til equity-loggen.
        """
        now = self._dt(ts)
        self._last_prices.update({s: float(p) for s, p in prices.items()})

        # Ny handelsdag? nulstil dagligt anker og tab-stop
        if now.date() != self.daily_anchor_date:
            self.daily_anchor_date = now.date()
            self.daily_start_equity = self._equity(self._last_prices)
            self.trading_halted = False  # ny dag → må handle igen

        # Fyld limit-ordrer hvis kursen er nået/passeret
        self._scan_limit_orders(now)

        # Snapshot af equity/drawdown
        equity = self._equity(self._last_prices)
        self.peak_equity = max(self.peak_equity, equity)
        dd_pct = 0.0 if self.peak_equity <= 0 else (equity - self.peak_equity) / self.peak_equity * 100.0

        self._append_equity_snapshot(now.date().isoformat(), equity, self.cash, self._positions_value(self._last_prices), dd_pct)

        # Tjek daily loss limit efter valuation
        self._check_daily_loss_limit(equity)

        return {
            "ts": now.isoformat(),
            "equity": equity,
            "cash": self.cash,
            "positions_value": self._positions_value(self._last_prices),
            "drawdown_pct": dd_pct,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "open_orders": [asdict(o) for o in self.open_orders],
            "trading_halted": self.trading_halted,
        }

    def pnl_snapshot(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Returnér kort PnL-oversigt. Opdaterer ikke logs."""
        px = prices or self._last_prices
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self._unrealized_pnl(px),
            "equity": self._equity(px),
            "cash": self.cash,
        }

    # ------- interne beregninger / fills -------

    def _get_last_price(self, symbol: str) -> Optional[float]:
        return self._last_prices.get(symbol)

    def _scan_limit_orders(self, now: datetime) -> None:
        """Gennemgå åbne limitordrer og fyld dem, hvor kursen er nået/passeret."""
        remaining: List[Order] = []
        for o in self.open_orders:
            last = self._get_last_price(o.symbol)
            if last is None:
                remaining.append(o)
                continue

            should_fill = (o.side == "BUY" and last <= float(o.limit_price)) or (o.side == "SELL" and last >= float(o.limit_price))
            if should_fill:
                self._try_fill_limit_now(o, last, now)
                if o.status != "filled":
                    # behold åben hvis ikke fyldt (fx pga. min_notional/cash/no-short)
                    remaining.append(o)
            else:
                remaining.append(o)
        self.open_orders = remaining

    def _limit_exec_raw_price(self, order: Order, last_px: float) -> float:
        """Rå eksekveringspris: BUY → min(last, limit), SELL → max(last, limit)."""
        if order.side == "BUY":
            return min(last_px, float(order.limit_price))
        return max(last_px, float(order.limit_price))

    def _try_fill_limit_now(self, order: Order, last_px: float, ts: Optional[datetime]) -> None:
        """
        Forsøg at fylde en limit-ordre, når touch-betingelsen er opfyldt.
        - Brug rå eksekveringspris (min/max af last og limit afh. af side).
        - min_notional valideres ved fill i _execute_fill (ikke ved submit, medmindre reject_below_min=True).
        - Kontant/no-short tjek håndteres i _execute_fill.
        """
        raw_exec = self._limit_exec_raw_price(order, last_px)

        # Prøv at eksekvere; _execute_fill kan evt. afvise pga. min_notional/min_qty/cash/no-short
        self._execute_fill(order, raw_exec, ts=ts)
        if order.status != "rejected":
            order.status = "filled"
            order.reason = None
        else:
            # hvis afvist ved fill (fx under min_notional), behold som åben ordre
            order.status = "open"
            order.reason = None  # åben ordre bærer ikke reject-reason

    def _execute_fill(self, order: Order, raw_price: float, ts: Optional[datetime] = None) -> None:
        """
        Udfør en handel til pris med slippage/commission, opdater konti/positioner, log fill.

        Min-filters her:
          - min_qty: check efter afrunding (market + limit).
          - min_notional: check for begge ordretyper mod **rå pris** (uden slippage).
            (LIMIT kan være 'udskudt' til fill, ikke fravalgt).
        """
        ts_str = self._fmt_ts(ts)  # barens timestamp (UTC Z)
        side_mult = 1.0 if order.side == "BUY" else -1.0

        # --- pris til min_notional check: rå pris (uden slippage)
        check_price = float(raw_price)

        # --- slippage-pris til bogføring/cash/PnL
        slip = self.slippage_bp / 10_000.0
        exec_price = raw_price * (1.0 + slip) if order.side == "BUY" else raw_price * (1.0 - slip)
        exec_price = self._r(exec_price, self.price_decimals)

        qty = self._r(order.qty, self.qty_decimals)

        # --- min_qty efter afrunding
        if self.min_qty > 0.0 and abs(qty) + 1e-12 < self.min_qty:
            order.status = "rejected"
            order.reason = "Ordre under min_qty"
            return

        # --- min_notional ved fill mod rå pris
        if self.min_notional > 0.0 and abs(qty * check_price) + 1e-12 < self.min_notional:
            order.status = "rejected"
            order.reason = "Ordre under min_notional"
            return

        # Affordability for BUY (inkl. commission) — regnes mod exec_price
        if order.side == "BUY" and not self.allow_short:
            per_unit_total = exec_price * (1.0 + self.commission_bp / 10_000.0)
            max_affordable_qty = self.cash / per_unit_total if per_unit_total > 0 else 0.0
            if max_affordable_qty <= 1e-12:
                order.status = "rejected"
                order.reason = "Ikke nok kontantdækning"
                return
            if qty - max_affordable_qty > 1e-12:
                order.status = "rejected"
                order.reason = f"Qty overstiger kontantdækning (max ~ {self._r(max_affordable_qty, self.qty_decimals)})"
                return

        # Ingen shorting: SELL må ikke skabe negativ netto-qty
        if not self.allow_short and order.side == "SELL":
            pos_now = self.positions.get(order.symbol) or Position(symbol=order.symbol, qty=0.0, avg_price=0.0)
            if pos_now.qty - qty < -1e-12:
                order.status = "rejected"
                order.reason = "Shorting ikke tilladt (ville gå under 0)"
                return

        notional = exec_price * qty
        commission = abs(notional) * (self.commission_bp / 10_000.0)

        # Opdater kontanter
        if order.side == "BUY":
            self.cash -= notional + commission
        else:
            self.cash += notional - commission

        # Opdater position + realiseret PnL
        pos = self.positions.get(order.symbol) or Position(symbol=order.symbol, qty=0.0, avg_price=0.0)
        realized = 0.0

        if pos.qty == 0.0 or math.copysign(1, pos.qty) == math.copysign(1, side_mult):
            # Samme retning (tilføj)
            new_qty = pos.qty + side_mult * qty
            if abs(new_qty) > 1e-12:
                pos.avg_price = (abs(pos.qty) * pos.avg_price + qty * exec_price) / abs(new_qty)
            pos.qty = new_qty
        else:
            # Modsat retning → luk helt/delvist og evt. flip
            closing_qty = min(abs(pos.qty), qty)
            if pos.qty > 0:  # lukker long med SELL
                realized += (exec_price - pos.avg_price) * closing_qty
            else:            # lukker short med BUY
                realized += (pos.avg_price - exec_price) * closing_qty

            remaining = qty - closing_qty
            pos.qty = math.copysign(abs(pos.qty) - closing_qty, pos.qty)

            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
                pos.avg_price = 0.0

            # Hvis der er resterende mængde, åbnes ny position i handelsretningen
            if remaining > 1e-12:
                if not self.allow_short and side_mult < 0:
                    # ville flippe til short → afvis resten (den lukkende del er allerede håndteret)
                    pass
                else:
                    new_qty = side_mult * remaining
                    pos.avg_price = exec_price
                    pos.qty = new_qty

        self.positions[order.symbol] = pos
        self.realized_pnl += realized

        # Log fill
        self._append_fill(Fill(
            ts=ts_str,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=exec_price,
            commission=commission,
            pnl_realized=realized,
        ))

        # Efter fill → tjek daglig loss limit ift. equity
        self._check_daily_loss_limit(self._equity(self._last_prices))

    # ------- PnL/Equity helpers -------

    def _positions_value(self, prices: Dict[str, float]) -> float:
        """Netto markedsværdi af positioner (qty * last)."""
        val = 0.0
        for s, pos in self.positions.items():
            px = prices.get(s)
            if px is not None:
                val += pos.qty * px
        return val

    def _unrealized_pnl(self, prices: Dict[str, float]) -> float:
        pnl = 0.0
        for s, pos in self.positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            px = prices.get(s)
            if px is None:
                continue
            if pos.qty > 0:
                pnl += (px - pos.avg_price) * abs(pos.qty)
            else:
                pnl += (pos.avg_price - px) * abs(pos.qty)
        return pnl

    def _equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self._positions_value(prices)

    # ------- risiko / loss limit -------

    def _check_daily_loss_limit(self, equity: float) -> None:
        """Stop ny handel hvis dagens tab overskrider grænsen (robust mod floating point)."""
        if self.daily_loss_limit_pct <= 0:
            return
        base = self.daily_start_equity
        if base <= 0:
            return
        loss_amt = max(0.0, base - equity)
        limit_amt = base * (abs(self.daily_loss_limit_pct) / 100.0)
        eps = max(1e-9, base * 1e-12)
        if loss_amt >= limit_amt - eps:
            self.trading_halted = True

    # ------- tidsformat -------

    def _fmt_ts(self, ts: Optional[datetime]) -> str:
        if ts is None:
            ts = self._now()
        elif ts.tzinfo is None:
            ts = self.tz.localize(ts)
        return ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _dt(self, ts: Optional[datetime]) -> datetime:
        if ts is None:
            return self._now()
        if ts.tzinfo is None:
            return self.tz.localize(ts)
        return ts.astimezone(self.tz)
