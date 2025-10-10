# -*- coding: utf-8 -*-
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# --- Sørg for at projektroden er på import-stien (stabil på tværs af OS/CI) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# Pytest hooks & options
# =========================


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Tilføj flag for at aktivere kontrakt-tests lokalt/CI:
      pytest --run-contract
    Alternativt: RUN_CONTRACT=1 pytest
    """
    parser.addoption(
        "--run-contract",
        action="store_true",
        default=False,
        help="Run tests marked as 'contract' (default: skipped).",
    )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Skip 'contract' tests med mindre de er eksplicit aktiveret.
    """
    run_contract = config.getoption("--run-contract") or _env_flag("RUN_CONTRACT")
    if run_contract:
        return

    skip_contract = pytest.mark.skip(
        reason="Skipping @contract tests (enable with --run-contract or RUN_CONTRACT=1)."
    )
    for item in items:
        if "contract" in item.keywords:
            item.add_marker(skip_contract)


# =========================
# Metrics server auto-start
# =========================


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def pytest_sessionstart(session) -> None:
    """
    Starter en minimal FastAPI-/uvicorn-proces for /metrics, så
    tests/test_metrics_exposition.py kan ramme http://localhost:8000/metrics.

    Styring via env:
      START_METRICS_SERVER=0  -> disable
      METRICS_PORT=8081       -> port override
    """
    if os.environ.get("START_METRICS_SERVER", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    port = int(os.environ.get("METRICS_PORT", "8000"))
    if not _port_free(port):
        # Allerede nogen der lytter (fx separat CI-step) -> gør intet
        return

    def _run():
        # Importér uvicorn først i child-thread for at undgå overhead i collection-fase
        import uvicorn  # type: ignore

        # Peg på din eksisterende app med metrics-route i api/app.py
        uvicorn.run("api.app:app", host="0.0.0.0", port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # kort spin-up buffer (stabil mod flakiness i hurtige CI-miljøer)
    time.sleep(0.7)


# =========================
# Generelle, delte fixtures
# =========================


@pytest.fixture(scope="session", autouse=True)
def _prepare_test_env() -> None:
    """
    Headless plotting + sikre artefaktmapper findes.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Sørg for at rapportmapper findes (pytest.ini skriver til disse)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("htmlcov").mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def _set_global_seed() -> int:
    """
    Global determinisme i tests (rng, np, evt. frameworks senere).
    Returnerer seed hvis man vil logge den i tests.
    """
    seed = 42
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    return seed


@pytest.fixture
def sample_config():
    """
    Minimal gyldig config til config-valideringstests.
    Matcher kravene i utils/config_utils.validate_config.
    """
    return {
        "strategies": ["mean_reversion"],
        "data": {"paths": {"raw": "/data/raw", "processed": "/data/processed"}},
        "trading": {"risk": {"max_position": 0.2}},
    }


@pytest.fixture
def dummy_returns():
    """Simple afkastsekvens til Sharpe/Sortino/drawdown-tests (deterministisk)."""
    return [0.01, -0.02, 0.03, 0.02, -0.01]


@pytest.fixture
def dummy_balance():
    """Balance-sekvens til drawdown-tests (deterministisk)."""
    return [100, 120, 80, 130, 110]


@pytest.fixture
def dummy_features_df():
    """
    Repræsentativt features-DataFrame til feature-/modeltests.
    - Deterministisk via global seed
    - Bruger 'H' for at undgå FutureWarning i pandas
    """
    n = 10
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="H"),
            "close": np.linspace(100, 110, n),
            "signal": np.random.choice([1, 0, -1], size=n),
            "ema_9": np.linspace(100, 105, n),
            "ema_21": np.linspace(100, 108, n),
            "ema_50": np.linspace(100, 109, n),
            "ema_200": np.linspace(100, 110, n),
            "rsi_14": np.random.uniform(30, 70, n),
            "macd": np.random.uniform(-2, 2, n),
            "macd_signal": np.random.uniform(-2, 2, n),
            "atr_14": np.random.uniform(0.5, 2, n),
            "regime": np.random.choice(["bull", "bear"], size=n),
        }
    )
    # Fastlås de første signaler, så tests kan forvente bestemte cases
    df.loc[0, "signal"] = 1
    df.loc[1, "signal"] = -1
    return df


@pytest.fixture
def dummy_preds():
    """Preds til ensemble-voting-tests (deterministisk)."""
    return [1, 0, 1, -1, 0, 1]


# ============================================
# Integrationstest- & GUI-hjælpefixtures
# ============================================


@pytest.fixture(scope="session")
def dummy_csv_path(tmp_path_factory) -> str:
    """
    Laver en midlertidig OHLCV-dummy CSV til pipeline/integrationstests.
    Hvis du vil teste med din egen fil: overskriv denne fixture i testmodulet.
    """
    tmpdir = tmp_path_factory.mktemp("data")
    dst = tmpdir / "ohlcv.csv"

    # Generér en lille, men realistisk, time-series
    periods = 50
    ts = pd.date_range("2025-01-01", periods=periods, freq="H")
    base = 100.0

    # Simpel prisbane med små variationer
    close = base + np.cumsum(np.random.normal(0, 0.2, size=periods)).round(2)
    high = (close + np.abs(np.random.normal(0.2, 0.05, size=periods))).round(2)
    low = (close - np.abs(np.random.normal(0.2, 0.05, size=periods))).round(2)
    open_ = (close + np.random.normal(0, 0.1, size=periods)).round(2)
    vol = (np.random.uniform(8_000, 12_000, size=periods)).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": np.maximum.reduce([open_, high, close]),
            "low": np.minimum.reduce([open_, low, close]),
            "close": close,
            "volume": vol,
        }
    )
    df.to_csv(dst, index=False)
    return str(dst)


@pytest.fixture
def clean_outputs(tmp_path, monkeypatch) -> Dict[str, Path]:
    """
    Isolerer arbejdsmapper for integrationstests:
      - CWD peger på en frisk temp-rodsmappe
      - 'outputs/' og 'backups/' findes og er tomme
    Returnerer paths, så testen kan skrive artefakter dertil.
    """
    root = tmp_path
    outputs = root / "outputs"
    backups = root / "backups"
    reports = root / "reports"
    outputs.mkdir(exist_ok=True)
    backups.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    # Skift working directory, så run.py/engine skriver lokalt
    monkeypatch.chdir(root)

    # Evt. miljøflag for at indikere test-mode i din pipeline
    monkeypatch.setenv("BOT_ENV", "TEST")

    return {"root": root, "outputs": outputs, "backups": backups, "reports": reports}


@pytest.fixture
def timestamp_regex():
    """
    Regex til at matche timestamps i mappenavne/filnavne.
    Eksempler: 2025-08-23_14-30-59 eller 2025-08-23-14-30-59
    """
    return re.compile(r"\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2}")


# ==========================
# Små hjælpefunktioner
# ==========================


@pytest.fixture
def require_columns():
    """
    Hjælper til hurtigt at tjekke kolonnekrav i DataFrames i tests.
    Brug: require_columns(df, {"timestamp","close"})
    """

    def _check(df: pd.DataFrame, cols):
        missing = set(cols) - set(df.columns)
        assert not missing, f"Manglende kolonner: {missing}"

    return _check
