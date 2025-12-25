# tests/backtest/test_flagship_trend_v1_parity.py

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.strategies.flagship_trend_v1 import (
    FlagshipTrendConfig,
    FlagshipTrendV1Strategy,
    Signal,
)

# ... [ALT DET ANDRE I FILEN UÆNDRET] ...


@pytest.mark.heavy
def test_flagship_trend_v1_flagship_backtest_vs_paper_parity_loose():
    """
    End-to-end paritetstest for selve Flagship Trend v1.

    Kører kun, hvis CLI-skriptet scripts/run_backtest_flagship_v1.py findes.
    Ellers markeres testen som SKIPPED (feature ikke implementeret endnu).
    """
    symbol = "BTCUSDT"
    interval = "1h"
    tag = "dev1"

    # Hvis CLI ikke findes endnu → skip i stedet for FAIL
    if not Path("scripts/run_backtest_flagship_v1.py").exists():
        pytest.skip(
            "Flagship backtest CLI (scripts/run_backtest_flagship_v1.py) er ikke implementeret endnu."
        )

    # 1) Hent rå data (lokalt, men via Binance-API)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fetch_raw_ohlcv_binance",
            "--symbol",
            symbol,
            "--interval",
            interval,
            "--limit",
            "2000",
        ],
        check=True,
    )

    # 2) Kør Flagship-backtest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_backtest_flagship_v1",
            "--symbol",
            symbol,
            "--interval",
            interval,
            "--tag",
            tag,
            "--no-persist",
        ],
        check=True,
    )

    # 3) Kør Flagship-paper
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_paper_flagship_v1",
            "--symbol",
            symbol,
            "--interval",
            interval,
            "--tag",
            tag,
        ],
        check=True,
    )

    bt_path = Path(f"outputs/backtests/flagship_{symbol.lower()}_{interval}_{tag}.json")
    paper_path = Path(f"outputs/paper/flagship_{symbol.lower()}_{interval}_{tag}.json")

    assert bt_path.exists(), f"Mangler Flagship backtest-metrics: {bt_path}"
    assert paper_path.exists(), f"Mangler Flagship paper-metrics: {paper_path}"

    with bt_path.open("r", encoding="utf-8") as f:
        bt = json.load(f)
    with paper_path.open("r", encoding="utf-8") as f:
        paper = json.load(f)

    bt_trades = float(bt.get("num_trades", 0))
    paper_trades = float(paper.get("num_trades", 0))

    assert bt_trades > 0, "Backtest Flagship har ingen trades"
    assert paper_trades > 0, "Paper Flagship har ingen trades"

    trades_diff = abs(bt_trades - paper_trades)
    trades_tol = max(5.0, 0.30 * max(bt_trades, 1.0))
    assert trades_diff <= trades_tol, (
        f"For stor forskel i num_trades: backtest={bt_trades}, "
        f"paper={paper_trades}, diff={trades_diff}, tol={trades_tol}"
    )

    bt_profit = float(bt.get("profit_pct", 0.0))
    paper_profit = float(paper.get("profit_pct", 0.0))
    profit_diff = abs(bt_profit - paper_profit)
    profit_tol = max(10.0, 0.30 * max(abs(bt_profit), 1.0))

    # Kun kræv samme tegn, hvis begge er tydeligt væk fra 0
    sign_threshold = 5.0
    if abs(bt_profit) >= sign_threshold and abs(paper_profit) >= sign_threshold:
        assert (bt_profit >= 0) == (
            paper_profit >= 0
        ), f"Profit-tegn uenige: backtest={bt_profit}, paper={paper_profit}"

    assert profit_diff <= profit_tol, (
        f"For stor forskel i profit_pct: backtest={bt_profit}, "
        f"paper={paper_profit}, diff={profit_diff}, tol={profit_tol}"
    )
