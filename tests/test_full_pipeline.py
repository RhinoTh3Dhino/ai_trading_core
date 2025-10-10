# tests/test_full_pipeline.py
# -*- coding: utf-8 -*-
"""
E2E-integrationstest for hele trading-pipelinen – matcher Dag 3-målet:
- Kører fuld pipeline med dummy CSV (fra conftest.py's fixture)
- Skriver artefakter til isoleret temp 'outputs/' og 'backups/'
- Validerer signals.csv + portfolio_metrics.json + backup-mappe med timestamp

Kør:
    pytest -m e2e -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import pytest

# Sørg for adgang til projektrod (til evt. modulimport)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------
# Hjælpere
# ---------------------------


def _find_engine_entrypoint() -> Tuple[Callable[..., Any], str]:
    """
    Find en egnet entrypoint-funktion til at køre pipelinen direkte i-process:
      - engine.run_pipeline(...)
      - bot.engine.run_pipeline(...)
      - app.engine.run_pipeline(...)
      - src.engine.run_pipeline(...)
    Falder tilbage til:
      - engine.run_full_pipeline / .run
    Returnerer (callable, kvalificeret_navn) eller skipper testen venligt.
    """
    mod_candidates = ["engine", "bot.engine", "app.engine", "src.engine", "core.engine"]
    fn_candidates = ["run_pipeline", "run_full_pipeline", "run"]
    errors = []
    for mn in mod_candidates:
        try:
            mod = __import__(mn, fromlist=["*"])
        except Exception as e:
            errors.append(f"import {mn}: {e.__class__.__name__}: {e}")
            continue
        for fn in fn_candidates:
            cb = getattr(mod, fn, None)
            if callable(cb):
                return cb, f"{mn}.{fn}"
            else:
                errors.append(f"{mn}: fandt ikke callable '{fn}'")
    # Ingen egnet entrypoint fundet → skip med vejledning
    pytest.skip(
        "Fandt ikke en engine-entrypoint at kalde.\n"
        "Eksponér fx en af disse funktioner:\n"
        "  - engine.run_pipeline(data_path, outputs_dir, backups_dir, paper=True)\n"
        "  - bot.engine.run_pipeline(...)\n"
        "  - app.engine.run_pipeline(...)\n\n"
        "Forsøgte men fejlede:\n  - " + "\n  - ".join(errors)
    )


def _maybe_keys(d: dict, *alts) -> bool:
    """Returnér True hvis mindst én af nøglerne i alts findes i dict d."""
    return any(k in d for k in alts)


def _has_any_column(df: pd.DataFrame, cols) -> bool:
    return any(c in df.columns for c in cols)


# ---------------------------
# Tests
# ---------------------------


@pytest.mark.e2e
@pytest.mark.timeout(60)
def test_full_pipeline_genererer_outputs(
    dummy_csv_path, clean_outputs, timestamp_regex
):
    """
    Kør fuld pipeline via engine-entrypoint med:
      data_path=dummy_csv_path
      outputs_dir=clean_outputs['outputs']
      backups_dir=clean_outputs['backups']
      paper=True
    Valider artefakter og basale skemaer.
    """
    run_fn, fn_name = _find_engine_entrypoint()

    outputs_dir = str(clean_outputs["outputs"])
    backups_dir = str(clean_outputs["backups"])

    # Sæt "sikker" miljø (respekteres af de fleste pipelines)
    os.environ.setdefault("OFFLINE", "1")
    os.environ.setdefault("DRY_RUN", "1")
    os.environ.setdefault("TRADING_MODE", "paper")

    # Kør pipelinen (signatur forsøges fleksibelt)
    try:
        result = run_fn(  # type: ignore[misc]
            data_path=dummy_csv_path,
            outputs_dir=outputs_dir,
            backups_dir=backups_dir,
            paper=True,
        )
    except TypeError:
        # Fald tilbage på mere generiske signaturer
        try:
            result = run_fn(dummy_csv_path, outputs_dir, backups_dir, True)  # type: ignore[misc]
        except TypeError:
            # Sidste forsøg: kun data_path + outputs_dir
            result = run_fn(dummy_csv_path, outputs_dir)  # type: ignore[misc]

    # --- Valider artefakter i outputs/ ---
    signals_path = Path(outputs_dir) / "signals.csv"
    metrics_path = Path(outputs_dir) / "portfolio_metrics.json"

    assert signals_path.exists(), f"signals.csv mangler i {outputs_dir}"
    assert metrics_path.exists(), f"portfolio_metrics.json mangler i {outputs_dir}"

    # Basistjek af signals.csv
    sig = pd.read_csv(signals_path)
    assert len(sig) >= 1, "signals.csv er tom"
    assert _has_any_column(
        sig, ["timestamp", "datetime"]
    ), "signals.csv mangler timestamp/datetime"
    assert _has_any_column(
        sig, ["signal", "action"]
    ), "signals.csv mangler signal/action"

    # Basistjek af portfolio_metrics.json
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert isinstance(metrics, dict) and metrics, "portfolio_metrics.json er tom"
    assert _maybe_keys(metrics, "pnl", "profit_pct"), "metrics mangler pnl/profit_pct"
    assert _maybe_keys(
        metrics, "max_drawdown", "drawdown_pct"
    ), "metrics mangler drawdown"
    # Win-rate valgfri, men hvis til stede, skal den være tal
    if "win_rate" in metrics:
        assert isinstance(
            metrics["win_rate"], (int, float)
        ), "win_rate er ikke numerisk"

    # --- Backupmappe med timestamp ---
    backups = [d for d in Path(backups_dir).iterdir() if d.is_dir()]
    assert backups, f"Ingen backup-mappe oprettet i {backups_dir}"
    assert any(
        timestamp_regex.search(b.name) for b in backups
    ), "Backup-mappen bør indeholde timestamp i navnet"


@pytest.mark.e2e
@pytest.mark.timeout(60)
def test_full_pipeline_konsekvent_metrics_schema_ved_gentagelse(
    dummy_csv_path, clean_outputs
):
    """
    Kør pipelinen to gange og verificér at 'portfolio_metrics.json' eksisterer efter hver run
    samt at nøglefelter (pnl/drawdown) findes begge gange.
    (Vi kræver ikke identiske værdier – kun stabilt skema.)
    """
    run_fn, fn_name = _find_engine_entrypoint()

    outputs_dir = str(clean_outputs["outputs"])
    backups_dir = str(clean_outputs["backups"])

    os.environ.setdefault("OFFLINE", "1")
    os.environ.setdefault("DRY_RUN", "1")
    os.environ.setdefault("TRADING_MODE", "paper")

    def _run_once() -> dict:
        try:
            run_fn(  # type: ignore[misc]
                data_path=dummy_csv_path,
                outputs_dir=outputs_dir,
                backups_dir=backups_dir,
                paper=True,
            )
        except TypeError:
            try:
                run_fn(dummy_csv_path, outputs_dir, backups_dir, True)  # type: ignore[misc]
            except TypeError:
                run_fn(dummy_csv_path, outputs_dir)  # type: ignore[misc]

        metrics_path = Path(outputs_dir) / "portfolio_metrics.json"
        assert metrics_path.exists(), "portfolio_metrics.json mangler efter run"
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    m1 = _run_once()
    # Ryd kun metrics for at simulere 'nyt run' uden at fjerne andre artefakter
    (Path(outputs_dir) / "portfolio_metrics.json").unlink(missing_ok=True)
    m2 = _run_once()

    assert isinstance(m1, dict) and isinstance(m2, dict), "metrics skal være dict"
    assert _maybe_keys(m1, "pnl", "profit_pct") and _maybe_keys(
        m2, "pnl", "profit_pct"
    ), "pnl/profit_pct mangler"
    assert _maybe_keys(m1, "max_drawdown", "drawdown_pct") and _maybe_keys(
        m2, "max_drawdown", "drawdown_pct"
    ), "drawdown mangler"
