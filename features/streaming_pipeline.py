# features/streaming_pipeline.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

# Forventet Bar-model (fra data.schemas):
# Bar: venue, symbol, ts(ms), interval, open, high, low, close, volume, is_final: bool

EMA_FAST = 14
EMA_SLOW = 50
RSI_LEN  = 14
ATR_LEN  = 14

def _utc_date_from_ms(ms: int):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date()

@dataclass
class WilderAvg:
    length: int
    value: Optional[float] = None
    warm: int = 0

    def update(self, x: float) -> float:
        """Wilder's smoothing: avg = (prev*(n-1) + x) / n"""
        if self.value is None:
            self.value = x
            self.warm = 1
        else:
            self.value = (self.value * (self.length - 1) + x) / self.length
            self.warm = min(self.length, self.warm + 1)
        return self.value

    @property
    def is_warm(self) -> bool:
        return self.warm >= self.length

@dataclass
class EMA:
    length: int
    value: Optional[float] = None
    warm: int = 0

    def update(self, price: float) -> float:
        if self.value is None:
            # start med simpelt gennemsnit over de første n værdier
            # (vi "simulerer" ved at akkumulere til vi har n samples)
            # her bruger vi en blid init: første værdi = price,
            # og øger warm-indtil vi rammer n; efter n bruger vi alpha.
            self.value = price
            self.warm = 1
            return self.value
        alpha = 2.0 / (self.length + 1.0)
        self.value = (price - self.value) * alpha + self.value
        self.warm = min(self.length, self.warm + 1)
        return self.value

    @property
    def is_warm(self) -> bool:
        return self.warm >= self.length

@dataclass
class SymbolState:
    last_close: Optional[float] = None

    # EMA
    ema_fast: EMA = field(default_factory=lambda: EMA(EMA_FAST))
    ema_slow: EMA = field(default_factory=lambda: EMA(EMA_SLOW))

    # RSI via Wilder
    avg_gain: WilderAvg = field(default_factory=lambda: WilderAvg(RSI_LEN))
    avg_loss: WilderAvg = field(default_factory=lambda: WilderAvg(RSI_LEN))

    # ATR via Wilder
    atr_avg: WilderAvg = field(default_factory=lambda: WilderAvg(ATR_LEN))

    # VWAP (intra-dag / intradag): cum(P*V)/cum(V) pr. UTC-dag
    vwap_day: Optional[object] = None
    vwap_pv_cum: float = 0.0
    vwap_v_cum: float = 0.0

    # antal LUKKEDE bars set (til warmup-gate)
    closed_bars: int = 0

class StreamingFeaturePipeline:
    """
    Streaming-feature pipeline (MVP):
      - EMA(14), EMA(50)
      - RSI(14) (Wilder)
      - ATR(14) (Wilder, TR = max(H-L, |H-prevC|, |L-prevC|))
      - VWAP intradag, ankret pr. UTC-dag: cum((H+L+C)/3 * V) / cum(V)
    Kun lukkede barer (bar.is_final=True) opdaterer state og producerer features.
    """
    def __init__(self, min_warmup_bars: int = EMA_SLOW):
        # vi kræver mindst 50 lukkede bars før vi emitter features
        self.min_warmup_bars = max(EMA_SLOW, RSI_LEN, ATR_LEN)
        self.state: Dict[str, SymbolState] = {}

    def _get_state(self, key: str) -> SymbolState:
        if key not in self.state:
            self.state[key] = SymbolState()
        return self.state[key]

    def update(self, bar) -> Optional[Dict[str, float]]:
        """Returnér dict med features hvis bar er lukket og warmup er opfyldt; ellers None."""
        if not getattr(bar, "is_final", False):
            return None  # kun lukkede barer medtages

        st = self._get_state(f"{bar.symbol}|{bar.interval}")

        # --- VWAP intradag (ankret pr. UTC-dag) --------------------------------
        d = _utc_date_from_ms(int(bar.ts))
        if st.vwap_day != d:
            st.vwap_day = d
            st.vwap_pv_cum = 0.0
            st.vwap_v_cum = 0.0

        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        vol = float(bar.volume)
        st.vwap_pv_cum += typical * vol
        st.vwap_v_cum += vol
        vwap = (st.vwap_pv_cum / st.vwap_v_cum) if st.vwap_v_cum > 0 else float("nan")

        # --- EMA ----------------------------------------------------------------
        c = float(bar.close)
        st.ema_fast.update(c)
        st.ema_slow.update(c)

        # --- RSI (Wilder) -------------------------------------------------------
        rsi = float("nan")
        if st.last_close is not None:
            change = c - st.last_close
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            avg_gain = st.avg_gain.update(gain)
            avg_loss = st.avg_loss.update(loss)
            if st.avg_gain.is_warm and st.avg_loss.is_warm and avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            elif st.avg_gain.is_warm and st.avg_loss.is_warm and avg_loss == 0:
                rsi = 100.0
        # --- ATR (Wilder) -------------------------------------------------------
        atr = float("nan")
        if st.last_close is None:
            tr = float(bar.high) - float(bar.low)
        else:
            high = float(bar.high)
            low = float(bar.low)
            prev_c = float(st.last_close)
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))
        atr_val = st.atr_avg.update(tr)
        if st.atr_avg.is_warm:
            atr = atr_val

        # update last_close + closed bars count (til warmup-gate)
        st.last_close = c
        st.closed_bars += 1

        # warmup gate — producer først features når vi har nok lukkede bars
        if st.closed_bars < self.min_warmup_bars:
            return None

        # EMA’er kan være “semi-warm” – men vi kræver alligevel min_warmup_bars
        ema_14 = st.ema_fast.value
        ema_50 = st.ema_slow.value

        # sørg for at ingen NaN slipper ud (DoD: “ingen NaN-leaks”)
        def _clean(x: float) -> Optional[float]:
            return None if (x is None or isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)

        out = {
            "ema_14": _clean(ema_14),
            "ema_50": _clean(ema_50),
            "rsi_14": _clean(rsi),
            "vwap":   _clean(vwap),
            "atr_14": _clean(atr),
        }
        # hvis noget ikke er klar → returnér None (hellere droppe end at skrive NaN)
        if any(v is None for v in out.values()):
            return None
        return out
