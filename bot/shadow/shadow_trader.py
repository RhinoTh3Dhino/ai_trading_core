# bot/shadow/shadow_trader.py

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Strategy(Protocol):
    def on_bar(self, bar) -> None: ...


class TheoreticalFillEngine(Protocol):
    def simulate_fill(self, order) -> float:
        """Returner teoretisk fill price givet historisk/stream data."""
        ...


@dataclass
class TradePair:
    ts: dt.datetime
    symbol: str
    side: str
    qty: float
    price_real: float
    price_theo: float

    @property
    def te_abs(self) -> float:
        """Absolut tracking error i procent."""
        return abs(self.price_real - self.price_theo) / self.price_theo


class ShadowTrader:
    """
    Shadow-trader:
    - Lytter på samme data/strategi som live-trader.
    - Sender ingen ordrer til venue.
    - Beregner teoretiske fills via TheoreticalFillEngine.
    - Logger tracking error (TE) pr. trade og kan lave dagsrapport.
    """

    def __init__(
        self,
        strategy: Strategy,
        fill_engine: TheoreticalFillEngine,
        output_dir: str = "outputs",
    ) -> None:
        self._strategy = strategy
        self._fill_engine = fill_engine
        self._output_dir = Path(output_dir)
        self._pairs: list[TradePair] = []

    # ---------- Data-flow ----------

    def on_bar(self, bar) -> None:
        """
        Kaldes fra din live-loop på samme måde som din primære strategi.
        """
        self._strategy.on_bar(bar)

    def record_fill_pair(
        self,
        *,
        ts: dt.datetime,
        symbol: str,
        side: str,
        qty: float,
        price_real: float,
        order_obj,
    ) -> None:
        """
        Kaldes når en rigtig paper-fill sker.
        Bruger order_obj til at spørge fill_engine om teoretisk pris.
        """
        price_theo = self._fill_engine.simulate_fill(order_obj)

        self._pairs.append(
            TradePair(
                ts=ts,
                symbol=symbol,
                side=side,
                qty=qty,
                price_real=price_real,
                price_theo=price_theo,
            )
        )

    # ---------- Rapporter & metrics ----------

    def dump_daily_report(self, trading_day: dt.date) -> Path:
        """
        Gemmer CSV med alle trades og TE for den givne dag.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"paper_te_{trading_day.isoformat()}.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "symbol", "side", "qty", "price_real", "price_theo", "te_abs"])
            for p in self._pairs:
                writer.writerow(
                    [
                        p.ts.isoformat(),
                        p.symbol,
                        p.side,
                        p.qty,
                        f"{p.price_real:.8f}",
                        f"{p.price_theo:.8f}",
                        f"{p.te_abs:.8f}",
                    ]
                )

        return path

    def aggregate_metrics(self) -> dict:
        """
        Returnerer aggregerede TE-metrics til prometheus/Telegram:
        - n_trades
        - te_median
        - te_mean
        - te_max
        """
        if not self._pairs:
            return {"n_trades": 0}

        te_values = [p.te_abs for p in self._pairs]
        te_values_sorted = sorted(te_values)
        n = len(te_values_sorted)
        te_median = te_values_sorted[n // 2]
        te_mean = sum(te_values_sorted) / n

        return {
            "n_trades": n,
            "te_median": te_median,
            "te_mean": te_mean,
            "te_max": max(te_values_sorted),
        }
