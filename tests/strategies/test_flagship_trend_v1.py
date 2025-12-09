# tests/strategies/test_flagship_trend_v1.py

from datetime import datetime, timedelta

import pandas as pd

from bot.strategies.flagship_trend_v1 import (
    FlagshipTrendConfig,
    FlagshipTrendV1Strategy,
)


def make_dummy_ohlcv(n: int, start_price: float = 100.0) -> pd.DataFrame:
    """
    Enkel, stigende OHLCV-serie til tests.
    """
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    prices = [start_price + i for i in range(n)]

    data = {
        "timestamp": timestamps,
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [100.0 for _ in range(n)],
    }
    return pd.DataFrame(data)


def test_prepare_features_adds_expected_columns():
    cfg = FlagshipTrendConfig()
    strat = FlagshipTrendV1Strategy(cfg)

    df = make_dummy_ohlcv(100)
    df_feat = strat.prepare_features(df)

    # Kolonner skal være til stede
    for col in ["ema_fast", "ema_slow", "atr", "vol_ratio"]:
        assert col in df_feat.columns, f"kolonne '{col}' mangler i feature-df"

    # Sidste rækker skal være uden NaNs
    tail = df_feat.tail(10)
    assert not tail["ema_fast"].isna().any()
    assert not tail["ema_slow"].isna().any()
    assert not tail["atr"].isna().any()
    assert not tail["vol_ratio"].isna().any()


def test_generate_signals_trend_up_and_vol_ok_sets_signal_one():
    cfg = FlagshipTrendConfig(
        ema_fast=2,
        ema_slow=5,
        atr_period=3,
        min_vol_ratio=0.0001,  # lav, så vol_ratio næsten altid er ok
    )
    strat = FlagshipTrendV1Strategy(cfg)

    df = make_dummy_ohlcv(50, start_price=100.0)  # stigende
    df_signals = strat.generate_signals(df)

    assert "signal" in df_signals.columns

    last_rows = df_signals.tail(5)
    assert (last_rows["signal"] == 1).any(), "forventer mindst én long-bias bar i stigende trend"


def test_generate_signals_low_vol_gives_zero_signal():
    cfg = FlagshipTrendConfig(
        ema_fast=2,
        ema_slow=5,
        atr_period=3,
        min_vol_ratio=10.0,  # urealistisk høj → vol_ratio er for lav
    )
    strat = FlagshipTrendV1Strategy(cfg)

    df = make_dummy_ohlcv(50, start_price=100.0)
    df["close"] = 100.0
    df["open"] = 100.0
    df["high"] = 100.1
    df["low"] = 99.9

    df_signals = strat.generate_signals(df)

    assert (
        df_signals["signal"] == 1
    ).sum() == 0, "forventer ingen signaler ved lav volatilitet og høj min_vol_ratio"
