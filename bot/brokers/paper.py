# bot/brokers/paper.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---------- Datamodeller ----------


@dataclass
class PositionState:
    mode: str = "FLAT"  # "FLAT" | "LONG"
    qty: float = 0.0  # aktuel beholdning (BTC)
    entry_price: float = 0.0  # gennemsnitlig indpris for aktiv position
    entry_commission: float = 0.0  # kommission betalt ved indgang
    cum_realized: float = 0.0  # akkumuleret realiseret PnL (inkl. alle kommissioner ved åbne/lukke)
    last_bar_ts: Optional[str] = None  # sidste bar (ISO) vi har behandlet (anti-duplikering)
    last_mtm_ts: Optional[str] = None  # sidste mark-to-market snapshot (ISO)


# ---------- Hjælpere ----------


def _iso_now() -> str:
    """UTC ISO8601 med sekund-opløsning, tz-naiv (fx '2025-09-03T22:00:00')."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _to_iso(ts: object) -> str:
    """Konverterer alt rimeligt til ISO-streng (UTC, tz-naiv)."""
    if isinstance(ts, str):
        return ts
    try:
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t):
            return _iso_now()
        return t.tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return _iso_now()


def _fmt_qty(q: float, decimals: int) -> str:
    return f"{q:.{decimals}f}"


def _fmt_price(p: float, decimals: int) -> str:
    return f"{p:.{decimals}f}"


# ---------- PaperBroker ----------


class PaperBroker:
    """
    Minimal, deterministisk papir-broker til MVP/live-paper:

    - Åbner/lukker LONG baseret på binært signal (1=BUY, 0=SELL) i 'onepos'-mode.
    - Skriver fills i logs/fills.csv (append med header auto).
    - Skriver equity-snapshots i logs/equity.csv:
        equity = STARTING_EQUITY + cum_realized + unrealized
      hvor unrealized = (last_price - entry_price) * qty (kun når LONG).
    - Ved åbning fratrækkes entry-kommission straks fra cum_realized (omkostning).
    - Ved lukning lægges netto realiseret PnL (brutto - kommissioner) til cum_realized.
    - Anti-duplikering pr. bar via state.last_bar_ts.
    - Valgfri mark-to-market snapshot på hver bar via PAPER_EQUITY_MTM=1.

    Miljøvariabler (fornuftige defaults):
      STARTING_EQUITY       (float, default 10000)
      QTY_USD               (float, default 100)        # $-størrelse pr. åbning
      PAPER_FEE_BPS         (float, default 4)          # 0.04% per fill
      PAPER_SLIPPAGE_BPS    (float, default 1)          # 0.01% glidning per fill
      POSITION_MODE         (str,   default "onepos")   # kun "onepos" understøttes i MVP
      MIN_BARS_BETWEEN_FLIPS(int,   default 1)          # 1 = mindst én ny bar før flip
      PAPER_EQUITY_MTM      (0/1,   default 1)          # skriv equity snapshot på hver bar
      PAPER_DECIMALS_PRICE  (int,   default 2)
      PAPER_DECIMALS_QTY    (int,   default 6)
    """

    def __init__(self, logs_dir: Path, symbol: str):
        self.logs = Path(logs_dir)
        self.logs.mkdir(parents=True, exist_ok=True)

        # Filstier
        self.fills_csv = self.logs / "fills.csv"
        self.equity_csv = self.logs / "equity.csv"
        self.state_file = self.logs / ".paper_state.json"

        # Symbol (CSV-format uden '/')
        self.symbol = symbol.replace("/", "")

        # Konfiguration
        self.starting_equity = float(os.getenv("STARTING_EQUITY", "10000"))
        self.qty_usd = float(os.getenv("QTY_USD", "100"))
        self.fee_bps = float(os.getenv("PAPER_FEE_BPS", "4"))
        self.slip_bps = float(os.getenv("PAPER_SLIPPAGE_BPS", "1"))
        self.pos_mode = os.getenv("POSITION_MODE", "onepos").lower()
        self.min_flip = int(os.getenv("MIN_BARS_BETWEEN_FLIPS", "1"))
        self.use_mtm = os.getenv("PAPER_EQUITY_MTM", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self.dec_price = int(os.getenv("PAPER_DECIMALS_PRICE", "2"))
        self.dec_qty = int(os.getenv("PAPER_DECIMALS_QTY", "6"))

        # Tilstand
        self._state = self._load_state()

    # ---------- Persistens ----------

    def _load_state(self) -> PositionState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return PositionState(**data)
            except Exception:
                pass
        return PositionState()

    def _save_state(self) -> None:
        self.state_file.write_text(
            json.dumps(asdict(self._state), ensure_ascii=False),
            encoding="utf-8",
        )

    # ---------- CSV Append ----------

    def _append_csv(self, path: Path, header: list[str], row: list[str]) -> None:
        new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            if new_file:
                f.write(",".join(header) + "\n")
            f.write(",".join(row) + "\n")

    # ---------- Kommission & Slippage ----------

    def _commission(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 10000.0)

    def _exec_price(self, ref_price: float, side: str) -> float:
        """Returnér 'handlet' pris med slippage indregnet."""
        bp = self.slip_bps / 10000.0
        if side.upper() == "BUY":
            return ref_price * (1.0 + bp)
        return ref_price * (1.0 - bp)

    # ---------- Equity ----------

    def _read_equity_tail(self) -> tuple[float, float]:
        """
        Returnér (last_equity, rolling_max) fra equity.csv,
        eller (starting_equity, starting_equity) hvis fil ikke findes.
        """
        if self.equity_csv.exists():
            try:
                df = pd.read_csv(self.equity_csv)
                if not df.empty and "equity" in df.columns:
                    last_eq = float(df["equity"].iloc[-1])
                    roll_max = float(df["equity"].cummax().iloc[-1])
                    return last_eq, roll_max
            except Exception:
                pass
        return self.starting_equity, self.starting_equity

    def _compute_equity_now(self, last_price: float) -> float:
        """
        Beregn øjeblikkelig equity = starting + cum_realized + unrealized (hvis LONG).
        Mark-to-market bruger *rå* pris (ingen slippage).
        """
        eq = self.starting_equity + float(self._state.cum_realized)
        if self._state.mode == "LONG" and self._state.qty > 0 and last_price > 0:
            eq += (last_price - self._state.entry_price) * self._state.qty
        return eq

    def _append_equity_snapshot(self, ts_iso: str, equity_value: float) -> None:
        last_eq, roll_max = self._read_equity_tail()
        cur_eq = float(equity_value)
        roll_max = max(roll_max, cur_eq)
        dd_pct = (cur_eq / roll_max - 1.0) * 100.0 if roll_max > 0 else 0.0

        self._append_csv(
            self.equity_csv,
            header=["date", "equity", "drawdown_pct"],
            row=[ts_iso, _fmt_price(cur_eq, 2), _fmt_price(dd_pct, 2)],
        )

    # ---------- Offentlig API ----------

    def exec_signal(self, signal: int, price: float, ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Udfør ordrelogik baseret på signal og skriv fills/equity.
        signal: 1=BUY, 0=SELL
        price: bar-close/last pris (float)
        ts: ISO-tid for bar (valgfri; hvis None bruges nu)
        Returnerer et lille resultat-dict for debugging/metrics.
        """
        ts_iso = _to_iso(ts)
        opened = closed = False

        # Anti-duplikering pr. bar
        if self._state.last_bar_ts and self._state.last_bar_ts == ts_iso:
            # Valgfri MTM-snapshot selv hvis vi ikke åbner/lukker
            if self.use_mtm:
                eq_now = self._compute_equity_now(price)
                if self._state.last_mtm_ts != ts_iso:
                    self._append_equity_snapshot(ts_iso, eq_now)
                    self._state.last_mtm_ts = ts_iso
                    self._save_state()
            return {"opened": opened, "closed": closed, "ts": ts_iso}

        if self.pos_mode != "onepos":
            # MVP: kun 'onepos' understøttes
            self._state.last_bar_ts = ts_iso
            self._save_state()
            return {"opened": opened, "closed": closed, "ts": ts_iso}

        # Eksekver efter simpel strategi: FLAT + BUY -> åbn, LONG + SELL -> luk
        if self._state.mode == "FLAT" and int(signal) == 1:
            opened = self._open_long(price, ts_iso)
        elif self._state.mode == "LONG" and int(signal) == 0:
            closed = self._close_long(price, ts_iso)

        # Mark-to-market snapshot (skriver equity selv hvis ingen fill)
        if self.use_mtm:
            eq_now = self._compute_equity_now(price)
            self._append_equity_snapshot(ts_iso, eq_now)
            self._state.last_mtm_ts = ts_iso

        # Opdater 'sidste bar' og persistér state
        self._state.last_bar_ts = ts_iso
        self._save_state()

        return {"opened": opened, "closed": closed, "ts": ts_iso}

    # ---------- Interne ordreoperationer ----------

    def _open_long(self, ref_price: float, ts_iso: str) -> bool:
        if ref_price <= 0:
            return False

        qty = self.qty_usd / ref_price
        if qty <= 0:
            return False

        # afrund qty for pæn CSV/GUI
        qty = float(_fmt_qty(qty, self.dec_qty))
        exec_px = self._exec_price(ref_price, "BUY")
        notional = qty * exec_px
        commission = self._commission(notional)

        # Kommission fratrækkes cum_realized ved åbning (omkostning nu)
        self._state.cum_realized -= commission

        # Skriv fill
        self._append_csv(
            self.fills_csv,
            header=[
                "ts",
                "symbol",
                "side",
                "qty",
                "price",
                "commission",
                "pnl_realized",
            ],
            row=[
                ts_iso,
                self.symbol,
                "BUY",
                _fmt_qty(qty, self.dec_qty),
                _fmt_price(exec_px, self.dec_price),
                _fmt_price(commission, 2),
                _fmt_price(0.0, 2),
            ],
        )

        # Opdater positions-tilstand
        self._state.mode = "LONG"
        self._state.qty = qty
        self._state.entry_price = exec_px
        self._state.entry_commission = commission
        return True

    def _close_long(self, ref_price: float, ts_iso: str) -> bool:
        if self._state.mode != "LONG" or self._state.qty <= 0:
            return False

        qty = self._state.qty
        exec_px = self._exec_price(ref_price, "SELL")
        notional = qty * exec_px
        commission = self._commission(notional)

        gross = (exec_px - self._state.entry_price) * qty
        pnl_realized = gross - self._state.entry_commission - commission

        # Læg netto realiseret PnL til cum_realized
        self._state.cum_realized += pnl_realized

        # Skriv fill
        self._append_csv(
            self.fills_csv,
            header=[
                "ts",
                "symbol",
                "side",
                "qty",
                "price",
                "commission",
                "pnl_realized",
            ],
            row=[
                ts_iso,
                self.symbol,
                "SELL",
                _fmt_qty(qty, self.dec_qty),
                _fmt_price(exec_px, self.dec_price),
                _fmt_price(commission, 2),
                _fmt_price(pnl_realized, 2),
            ],
        )

        # Luk position
        self._state.mode = "FLAT"
        self._state.qty = 0.0
        self._state.entry_price = 0.0
        self._state.entry_commission = 0.0
        return True

    # ---------- Service-funktioner ----------

    def reset_state(self) -> None:
        """Nulstil tilstand (berører ikke CSV’er). Bruges kun til tests."""
        self._state = PositionState()
        self._save_state()

    def status(self) -> Dict[str, Any]:
        """Returnér et lille status-dict (kan logges i live.py)."""
        return {
            "mode": self._state.mode,
            "qty": self._state.qty,
            "entry_price": self._state.entry_price,
            "cum_realized": self._state.cum_realized,
            "last_bar_ts": self._state.last_bar_ts,
            "last_mtm_ts": self._state.last_mtm_ts,
        }
