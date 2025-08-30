# utils/alerts.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Tuple

# -------------------------------------------------------------------------------------------------
# Konfiguration af tærskler
# -------------------------------------------------------------------------------------------------
@dataclass
class AlertThresholds:
    dd_pct: float = 10.0        # Alert når drawdown <= -dd_pct  (fx -10%)
    winrate_min: float = 45.0   # Alert når winrate < winrate_min
    profit_pct: float = 20.0    # Alert når samlet PnL% >= profit_pct
    cooldown_s: float = 1800.0  # Cooldown per alert-type i sekunder


# -------------------------------------------------------------------------------------------------
# AlertManager
#   - Kald on_equity(equity) ved hver bar/mark-to-market.
#   - Kald on_fill(pnl_value) ved fills (pnl_value>0 tæller som vundet trade).
#   - Kald evaluate_and_notify(send_fn) periodisk; send_fn(kind, text) udfører selve beskeden.
# -------------------------------------------------------------------------------------------------
class AlertManager:
    def __init__(self, th: AlertThresholds, allow_alerts: bool = True):
        self.allow = allow_alerts
        self.th = th
        self._last_alert_ts: Dict[str, float] = {}

        # Equity-tilstand
        self._start_equity: Optional[float] = None
        self._peak_equity: Optional[float] = None
        self._cur_equity: Optional[float] = None  # <- VIGTIG: aktuel equity (fix af tidligere bug)

        # Winrate-tilstand
        self._wins: int = 0
        self._trades: int = 0

    # -----------------------------
    # Integration helpers
    # -----------------------------
    def on_fill(self, pnl_value: Optional[float] = None) -> None:
        """Kald ved hver fill; pnl_value>0 tæller som win."""
        self._trades += 1
        if pnl_value is not None and pnl_value > 0:
            self._wins += 1

    def on_equity(self, eq_value: float) -> None:
        """Kald ved hver bar/status med ny equity-værdi."""
        # Sæt start-equity én gang
        if self._start_equity is None:
            self._start_equity = float(eq_value)

        # Opdater peak-equity
        if self._peak_equity is None or eq_value > self._peak_equity:
            self._peak_equity = float(eq_value)

        # Opdater AKTUEL equity (så drawdown/pnl beregnes korrekt)
        self._cur_equity = float(eq_value)

    # -----------------------------
    # Afledte metrikker
    # -----------------------------
    def _current_equity(self) -> Optional[float]:
        """Returnér aktuel equity (ikke peak)."""
        return self._cur_equity

    def _pnl_pct(self) -> Optional[float]:
        """Samlet PnL i % relativt til start equity."""
        if self._start_equity and self._start_equity != 0:
            cur = self._current_equity()
            if cur is not None:
                return (cur / self._start_equity - 1.0) * 100.0
        return None

    def _dd_pct(self) -> Optional[float]:
        """Aktuel drawdown i % relativt til peak equity (negativ ved tab fra peak)."""
        cur = self._current_equity()
        if cur is not None and self._peak_equity and self._peak_equity != 0:
            return (cur / self._peak_equity - 1.0) * 100.0
        return None

    def _winrate(self) -> Optional[float]:
        """Vundne handler i %."""
        if self._trades > 0:
            return (self._wins / self._trades) * 100.0
        return None

    # -----------------------------
    # Cooldown pr. alert-type
    # -----------------------------
    def _cooldown_ok(self, key: str) -> bool:
        """Returnér True hvis cooldown er udløbet for alert-typen 'key'."""
        now = time.monotonic()
        last = self._last_alert_ts.get(key, 0.0)
        if now - last >= self.th.cooldown_s:
            self._last_alert_ts[key] = now
            return True
        return False

    # -----------------------------
    # Evaluering & notifikation
    # -----------------------------
    def evaluate_and_notify(self, send_fn: Callable[[str, str], None]) -> None:
        """
        Kaldes periodisk (fx hver bar eller ved fill).
        Sender beskeder via send_fn(kind:str, text:str).
        """
        if not self.allow:
            return

        # Drawdown
        dd = self._dd_pct()
        if dd is not None and dd <= -abs(self.th.dd_pct):
            if self._cooldown_ok("dd"):
                send_fn("alert", f"🔻 Drawdown {dd:.2f}% (grænse {self.th.dd_pct:.2f}%)")

        # Winrate (kræver minimum antal handler for at undgå støj)
        wr = self._winrate()
        if wr is not None and self._trades >= 10 and wr < self.th.winrate_min:
            if self._cooldown_ok("winrate"):
                send_fn("alert", f"⚠️ Win-rate {wr:.1f}% under {self.th.winrate_min:.1f}% (trades={self._trades})")

        # Profit
        pnl = self._pnl_pct()
        if pnl is not None and pnl >= self.th.profit_pct:
            if self._cooldown_ok("profit"):
                send_fn("alert", f"✅ PnL {pnl:.2f}% over {self.th.profit_pct:.2f}% (take-profit signal?)")

    # -----------------------------
    # Hjælpere til tests/debug
    # -----------------------------
    def snapshot(self) -> dict:
        """Returnér et lille overblik til debugging/tests."""
        return {
            "start": self._start_equity,
            "peak": self._peak_equity,
            "current": self._cur_equity,
            "dd_pct": self._dd_pct(),
            "pnl_pct": self._pnl_pct(),
            "winrate": self._winrate(),
            "wins": self._wins,
            "trades": self._trades,
            "last_alert_ts": dict(self._last_alert_ts),
        }


# -------------------------------------------------------------------------------------------------
# Default sender (fallback til print hvis Telegram ikke er wired op)
# -------------------------------------------------------------------------------------------------
def default_send_fn(kind: str, text: str) -> None:
    """Bruger _tg_send hvis den findes; ellers print."""
    try:
        # _tg_send forventes fra engine-wrapper; ignorer hvis ikke defineret.
        _tg_send(kind, text)  # type: ignore[name-defined]
    except Exception:
        print(f"[{kind.upper()}] {text}")
