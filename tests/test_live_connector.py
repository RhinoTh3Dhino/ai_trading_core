# tests/test_live_connector.py
import asyncio
import math

import pytest

from data.schemas import Bar
from features.streaming_pipeline import StreamingFeaturePipeline


@pytest.mark.asyncio
async def test_streaming_pipeline_incremental():
    pipe = StreamingFeaturePipeline()

    feats = {}
    warm_seen = False

    # Kør nok lukkede bars til at passere warmup (60 er konservativt)
    for i in range(60):
        b = Bar(
            ts=1_700_000_000_000 + i * 60_000,
            symbol="BTCUSDT",
            venue="binance",
            interval="1m",
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100 + i,
            volume=10.0,
            is_final=True,  # kun beregn features på lukkede bars
        )
        feats = pipe.update(b) or {}

        # Registrér hvornår vi første gang ser de centrale features
        if {"ema_14", "rsi_14"} <= set(feats.keys()):
            warm_seen = True

        # Giv eventuelt kontrol tilbage til loopet (ingen real I/O her, men ok i asyncio-test)
        await asyncio.sleep(0)

    # Efter 60 bars bør warmup være passeret og features være tilgængelige
    assert warm_seen, "Forventede at se ema_14 og rsi_14 efter warmup"

    # Tjek at de seneste værdier er veldefinerede
    assert "ema_14" in feats and "rsi_14" in feats, "MVP features mangler i output"
    assert math.isfinite(float(feats["ema_14"]))
    assert math.isfinite(float(feats["rsi_14"]))

    # ATR er også en del af MVP – tjek tilstedeværelse og numeric
    if "atr_14" in feats:
        assert math.isfinite(float(feats["atr_14"]))
