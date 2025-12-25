# from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FlagshipTrendConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"

    # Risk
    risk_per_trade_pct: float = 0.01  # 1 % af equity pr. trade
    daily_loss_limit_pct: float = 0.03  # stop for dagen ved -3 %

    # Cooldown
    cooldown_bars_after_exit: int = 3

    # Trend- og entry-filtre
    fast_ema_col: str = "ema_21"
    slow_ema_col: str = "ema_50"
    rsi_col: str = "rsi_14"
    rsi_trend_min: float = 55.0

    min_pv_ratio: float = 0.5  # minimum pv_ratio
    min_volume: float = 0.0  # hårdt volume-gulv (kan sættes højere senere)

    # Volatilitet / SL / TP
    atr_col: str = "atr_14"
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0


@dataclass
class Signal:
    """
    Simpelt signal-objekt til brug i backtest/paper.

    Tilpas denne til din "rigtige" Signal-type, hvis du allerede har en anden
    definition i koden. Struktur:

    - symbol: fx "BTCUSDT"
    - side: "BUY" eller "SELL"
    - qty: handelsstørrelse (base-qty)
    - reason: tekst til debugging/metrics
    """

    symbol: str
    side: str  # "BUY" eller "SELL"
    qty: float
    reason: Optional[str] = None


class FlagshipTrendV1Strategy:
    """
    Enkel trend-following strategi:

    - Går LONG i optrend:
        * fast_ema > slow_ema
        * RSI over rsi_trend_min
        * volume/pv_ratio ok
    - SL/TP baseret på ATR (sl_atr_mult / tp_atr_mult)
    - Daily loss limit på equity
    - Cooldown i X bars efter exit
    """

    def __init__(self, cfg: FlagshipTrendConfig) -> None:
        self.cfg = cfg
        self._cooldown_bars_remaining: int = 0

    # ------------------------------------------------------------------ #
    # Internt state-hjælp                                                 #
    # ------------------------------------------------------------------ #

    def _tick_cooldown(self) -> None:
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1

    def _in_cooldown(self) -> bool:
        return self._cooldown_bars_remaining > 0

    def _start_cooldown(self) -> None:
        self._cooldown_bars_remaining = self.cfg.cooldown_bars_after_exit

    # ------------------------------------------------------------------ #
    # Hoved-API: on_bar                                                   #
    # ------------------------------------------------------------------ #

    def on_bar(
        self,
        bar: Any,
        account_state: dict,
        daily_pnl: float,
    ) -> Optional[Signal]:
        """
        Kaldes én gang pr. bar.

        Parameters
        ----------
        bar : fx pandas.Series eller dict
            Skal mindst have: close, high, low, volume, ema_21, ema_50, rsi_14, atr_14, pv_ratio.
        account_state : dict
            Forventet keys:
              - equity: samlet konto-værdi (float)
              - position_qty: nuværende qty (float, >0 hvis long)
              - position_side: fx "LONG" eller None
              - entry_price: gennemsnitlig entry-pris for åben position (float eller None)
        daily_pnl : float
            Dagens PnL i samme valuta som equity.

        Returns
        -------
        Optional[Signal]
            BUY/SELL signal eller None hvis ingen handling.
        """
        # 1) Cooldown tæller ned hver bar
        self._tick_cooldown()

        equity = float(account_state.get("equity", 0.0))
        position_qty = float(account_state.get("position_qty", 0.0))
        position_side = account_state.get("position_side")  # fx "LONG" eller None
        entry_price = account_state.get("entry_price")

        # 2) Hard daily loss limit – ingen nye trades, evt. force exit
        if equity > 0 and daily_pnl <= -self.cfg.daily_loss_limit_pct * equity:
            # Hvis vi har en åben position → luk den og gå i cooldown
            if position_qty != 0.0:
                # Vi antager kun LONG i v1 – SELL lukker positionen
                side = "SELL" if position_side == "LONG" else "BUY"
                qty = abs(position_qty)
                self._start_cooldown()
                return Signal(
                    symbol=self.cfg.symbol,
                    side=side,
                    qty=qty,
                    reason="daily_loss_limit",
                )
            # Ingen position → bare blokér nye entries
            return None

        # Træk basale felter ud fra bar (Series eller dict)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar["volume"])

        fast_ema = float(bar[self.cfg.fast_ema_col])
        slow_ema = float(bar[self.cfg.slow_ema_col])
        rsi = float(bar[self.cfg.rsi_col])
        atr = float(bar.get(self.cfg.atr_col, 0.0))
        pv_ratio = float(bar.get("pv_ratio", 0.0))

        # ------------------------------------------------------------------
        # 3) Position management – hvis vi allerede er LONG
        # ------------------------------------------------------------------
        if position_qty > 0.0 and position_side == "LONG":
            exit_reason: Optional[str] = None

            # Trend-break: fast EMA krydser ned under slow EMA
            if fast_ema <= slow_ema:
                exit_reason = "trend_break"

            # ATR-baseret SL / TP hvis vi har entry_price og ATR > 0
            if entry_price is not None and atr > 0:
                sl = entry_price - self.cfg.sl_atr_mult * atr
                tp = entry_price + self.cfg.tp_atr_mult * atr

                if low <= sl:
                    exit_reason = exit_reason or "stop_loss"
                if high >= tp:
                    exit_reason = exit_reason or "take_profit"

            if exit_reason is not None:
                self._start_cooldown()
                return Signal(
                    symbol=self.cfg.symbol,
                    side="SELL",  # v1: kun long-strategi
                    qty=position_qty,
                    reason=exit_reason,
                )

            # Behold positionen
            return None

        # ------------------------------------------------------------------
        # 4) Flat – kan vi åbne ny LONG?
        # ------------------------------------------------------------------
        if self._in_cooldown():
            # Vi må ikke åbne ny position midt i cooldown
            return None

        # Trendfilter: optrend + "styrke"
        in_uptrend = fast_ema > slow_ema and rsi >= self.cfg.rsi_trend_min
        volume_ok = volume >= self.cfg.min_volume and pv_ratio >= self.cfg.min_pv_ratio

        if not (in_uptrend and volume_ok):
            return None

        # Position sizing: risikér risk_per_trade_pct * equity med ATR-SL
        if equity <= 0.0 or atr <= 0.0:
            return None

        risk_amount = equity * self.cfg.risk_per_trade_pct
        sl_distance = self.cfg.sl_atr_mult * atr
        if sl_distance <= 0:
            return None

        # Qty ~ hvor meget base-asset vi kan købe, hvis SL rammes
        qty = risk_amount / sl_distance / close
        if qty <= 0:
            return None

        return Signal(
            symbol=self.cfg.symbol,
            side="BUY",
            qty=qty,
            reason="trend_entry",
        )
