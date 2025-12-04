# bot/config/binance.py

"""
Config-loader til Binance spot TESTNET.

Læser API-nøgler og base-url fra environment-variabler og returnerer
en BinanceExecutionConfig, som bruges af BinanceTestnetExecutionAdapter.
"""

from __future__ import annotations

import os

from bot.execution.binance_testnet_adapter import BinanceExecutionConfig


def load_binance_testnet_config() -> BinanceExecutionConfig:
    """
    Loader Binance TESTNET konfiguration fra env-variabler.

    Krævede variabler:
        BINANCE_TESTNET_API_KEY
        BINANCE_TESTNET_API_SECRET
    """
    api_key = os.environ["BINANCE_TESTNET_API_KEY"].strip()
    api_secret = os.environ["BINANCE_TESTNET_API_SECRET"].strip()

    base_url = os.getenv("BINANCE_TESTNET_BASE_URL", "https://testnet.binance.vision").strip()

    recv_window = int(os.getenv("BINANCE_TESTNET_RECV_WINDOW", "5000"))
    timeout_sec = int(os.getenv("BINANCE_TESTNET_TIMEOUT_SEC", "10"))
    max_retries = int(os.getenv("BINANCE_TESTNET_MAX_RETRIES", "3"))
    retry_backoff_sec = float(os.getenv("BINANCE_TESTNET_RETRY_BACKOFF_SEC", "1.0"))

    return BinanceExecutionConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        recv_window=recv_window,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
    )
