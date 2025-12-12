# from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from bot.features.indicators import atr, ema  # tilpas sti hvis nødvendig


@dataclass
class FlagshipTrendConfig:
    """
    Konfiguration for Lyra BTCUSDT Trend V1 – SPRINT 1 version.

    Live-specifikke ting (daily loss limit, cooldown, osv.)
    bliver først brugt rigtigt i SPRINT 3.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = "15m"

    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14

    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0

    risk_per_trade: float = 0.005
    daily_loss_limit_pct: float = 0.03
    min_vol_ratio: float = 0.002
    cooldown_bars: int = 4


class FlagshipTrendV1Strategy:
    """
    Lyra BTCUSDT Trend V1 – strategi-kerne til backtest.

    SPRINT 1:
    - prepare_features(df) → EMA/ATR/vol_ratio
    - generate_signals(df) → kolonne 'signal' i {0,1}
    - on_bar(...) er kun stub (implementeres senere til live/paper).
    """

    def __init__(self, config: FlagshipTrendConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Batch-funktioner til backtest
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tilføj EMA/ATR/vol_ratio til et OHLCV-DataFrame.

        Forventet input-kolonner:
        ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        """
        df = df.copy()

        df["ema_fast"] = ema(df["close"], self.config.ema_fast)
        df["ema_slow"] = ema(df["close"], self.config.ema_slow)
        df["atr"] = atr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            period=self.config.atr_period,
        )

        df["vol_ratio"] = df["atr"] / df["close"]

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch-signal til backtest.

        Output:
            df med ekstra kolonne 'signal' i {0, 1}:
            - 1: long-bias (trend op + vol_ok)
            - 0: neutral
        """
        df = self.prepare_features(df)
        df["signal"] = 0

        cond_trend_up = df["ema_fast"] > df["ema_slow"]
        cond_vol_ok = df["vol_ratio"] > self.config.min_vol_ratio

        df.loc[cond_trend_up & cond_vol_ok, "signal"] = 1
        return df

    # ------------------------------------------------------------------
    # Live/paper entrypoint – implementeres først i SPRINT 3
    # ------------------------------------------------------------------

    def on_bar(self, bar: Any, account_state: Any, daily_pnl: float):
        """
        Stub til live/paper.

        I SPRINT 1 skal denne ikke bruges.
        Vi lader den eksplodere hvis nogen kalder den,
        så vi opdager det tidligt.
        """
        raise NotImplementedError("on_bar implementeres i SPRINT 3 (live/paper-logik).")
