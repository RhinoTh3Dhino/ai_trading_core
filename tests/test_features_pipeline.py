"""
tests/test_features_pipeline.py

Dækker features/features_pipeline.py med dummy DataFrames (hurtige, deterministic).
Fokus: branch coverage + kerne-API (generate_features, save_features, load_features).
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Sørg for at projektroden er i sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.features_pipeline import (
    generate_features,
    save_features,
    load_features,
)


# ---------------------------------------------------------------------
# Hjælpere
# ---------------------------------------------------------------------
def make_version_with_timestamp(version: str) -> str:
    ts = datetime.now().strftime("%Y%m%d")
    return f"{version}_{ts}"


def ensure_dir_exists(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_dummy_df(rows: int = 60) -> pd.DataFrame:
    """Opret et deterministisk dummy-DF med OHLCV + timestamp."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
        "open": np.linspace(100, 100 + rows - 1, rows),
        "high": np.linspace(101, 101 + rows - 1, rows),
        "low": np.linspace(99, 99 + rows - 1, rows),
        "close": np.linspace(100, 100 + rows - 1, rows),
        "volume": np.arange(1000, 1000 + rows),
    })


# ---------------------------------------------------------------------
# Kerne-test: pipeline + save/load (uden patterns for determinisme)
# ---------------------------------------------------------------------
def test_generate_features_pipeline_and_save_load(tmp_path, monkeypatch):
    """
    End-to-end: DF -> generate_features -> save_features -> load_features.
    Vi deaktiverer patterns for at undgå eksterne afhængigheder i kerne-testen.
    """
    raw_df = make_dummy_df(60)
    assert len(raw_df) > 0

    # Kør featuregenerering – patterns disabled for deterministisk output
    features_df = generate_features(raw_df, feature_config={
        "patterns_enabled": False,
        "coerce_timestamps": True,
        "dropna": True,
        "target_mode": "direction",
        "horizon": 1,
    })
    assert not features_df.isnull().values.any(), "Features indeholder NaN!"

    expected_cols_core = [
        "rsi_14", "rsi_28",
        "macd", "macd_signal",
        "ema_9", "ema_21", "ema_50",
        "atr_14", "vwap",
        "bb_upper", "bb_lower",
        "return", "pv_ratio", "regime"
    ]
    for col in expected_cols_core:
        assert col in features_df.columns, f"Feature mangler: {col}"

    # Hvis target findes, må den ikke have NaN
    if "target" in features_df.columns:
        assert not features_df["target"].isnull().any()

    # Gem i projektets outputs/feature_data (samme sti som load_features forventer)
    ensure_dir_exists(PROJECT_ROOT / "outputs" / "feature_data")
    version_ts = make_version_with_timestamp("test")
    feature_path = save_features(features_df, "BTC", "1h", version_ts)
    assert Path(feature_path).exists(), f"Featurefil blev ikke gemt: {feature_path}"

    # Load og valider (samme kolonner som før)
    loaded_df = load_features("BTC", "1h", version_prefix=version_ts)
    assert len(loaded_df) > 0, "Indlæst features er tom"
    for col in expected_cols_core:
        assert col in loaded_df.columns, f"[LOAD] Feature mangler: {col}"


# ---------------------------------------------------------------------
# Kolonnekrav og fejlscenarier
# ---------------------------------------------------------------------
def test_generate_features_with_missing_columns_raises():
    """Tester at generate_features kaster fejl ved manglende kolonner (fx volume)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "open": range(100, 105),
        "high": range(101, 106),
        "low": range(99, 104),
        "close": range(100, 105),
        # volume mangler med vilje
    })

    with pytest.raises(Exception) as ei:
        _ = generate_features(df)
    assert "Mangler kolonner" in str(ei.value) or "Mangler" in str(ei.value)


def test_generate_features_empty_df_raises():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        generate_features(df)


# ---------------------------------------------------------------------
# Timestamp mapping & coercion
# ---------------------------------------------------------------------
def test_timestamp_mapping_accepts_datetime_aliases():
    df = make_dummy_df(10).rename(columns={"timestamp": "datetime"})
    out = generate_features(df, feature_config={"patterns_enabled": False})
    assert "ema_9" in out.columns


def test_timestamp_invalid_values_raise_even_with_coerce():
    df = make_dummy_df(5)
    df.loc[2, "timestamp"] = "INVALID"
    with pytest.raises(ValueError):
        generate_features(df, feature_config={"coerce_timestamps": True, "patterns_enabled": False})


# ---------------------------------------------------------------------
# Include/Exclude + dropna + normalize
# ---------------------------------------------------------------------
def test_include_exclude_and_keep_labels():
    df = make_dummy_df(30)
    out = generate_features(df, feature_config={
        "patterns_enabled": False,
        "include": ["ema_9", "ema_21", "pv_ratio", "non_existing"],
        "exclude": ["pv_ratio"],  # skal fjernes igen
        "target_mode": "direction",
        "dropna": True,
    })
    # Kun ema_9, ema_21 (og target/regime hvis de findes)
    for must in ["ema_9", "ema_21"]:
        assert must in out.columns
    assert "pv_ratio" not in out.columns
    # target/regime bør bevares hvis de eksisterer
    assert "regime" in out.columns
    # target er kun til stede hvis ikke tidligere sat og mode != none
    # (her er den sat af pipelinen i direction-mode)
    assert "target" in out.columns


def test_dropna_removes_rows_when_nan_present():
    df = make_dummy_df(10)
    df.loc[0, "close"] = np.nan  # fremprovokér NaN i features
    out = generate_features(df, feature_config={"patterns_enabled": False, "dropna": True})
    assert len(out) < len(df), "dropna=True burde reducere antal rækker ved NaN"


def test_normalize_scales_numeric_0_1_and_skips_labels():
    df = make_dummy_df(25)
    out = generate_features(df, feature_config={
        "patterns_enabled": False,
        "normalize": True,
        "target_mode": "direction",
    })
    # target og regime må ikke normaliseres
    assert set(["target", "regime"]).issubset(out.columns)
    # Tjek at nogle numeriske kolonner er inden for [0,1]
    for col in ["ema_9", "ema_21", "ema_50", "vwap", "pv_ratio"]:
        assert col in out.columns
        colmin, colmax = out[col].min(), out[col].max()
        assert 0.0 - 1e-9 <= colmin <= 1.0 + 1e-9
        assert 0.0 - 1e-9 <= colmax <= 1.0 + 1e-9


# ---------------------------------------------------------------------
# Target modes
# ---------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["direction", "regression", "none"])
def test_target_modes(mode):
    df = make_dummy_df(40)
    out = generate_features(df, feature_config={
        "patterns_enabled": False,
        "target_mode": mode,
        "horizon": 2,
        "dropna": True,
    })
    if mode == "none":
        # Pipeline sætter ikke target, hvis ikke allerede til stede
        assert "target" not in out.columns
    elif mode == "direction":
        assert "target" in out.columns
        assert set(out["target"].unique()).issubset({0, 1})
    else:  # regression
        assert "target" in out.columns
        # Ingen NaN efter fillna(0.0)
        assert not out["target"].isna().any()


def test_invalid_target_mode_raises():
    df = make_dummy_df(20)
    with pytest.raises(ValueError):
        generate_features(df, feature_config={"patterns_enabled": False, "target_mode": "weird"})


# ---------------------------------------------------------------------
# Feature match validering + vol_spike rename
# ---------------------------------------------------------------------
def test_expected_features_ok_and_fail():
    df = make_dummy_df(30)
    out = generate_features(df, feature_config={"patterns_enabled": False})
    # OK: kræv et subset der eksisterer
    ok_subset = ["ema_9", "ema_21", "macd"]
    out2 = generate_features(df, feature_config={
        "patterns_enabled": False,
        "expected_features": ok_subset
    })
    assert set(ok_subset).issubset(out2.columns)

    # FAIL: kræv en feature der ikke findes
    with pytest.raises(ValueError):
        _ = generate_features(df, feature_config={
            "patterns_enabled": False,
            "expected_features": ["totally_unknown_feature"]
        })


def test_vol_spike_is_renamed_to_volume_spike_when_present():
    # Tilføj manuelt 'vol_spike' i input – patterns disabled, så renaming sker kun pga. input
    df = make_dummy_df(15)
    df["vol_spike"] = 0
    out = generate_features(df, feature_config={"patterns_enabled": False})
    assert "volume_spike" in out.columns
    assert "vol_spike" not in out.columns


# ---------------------------------------------------------------------
# Valgfri sanity med patterns_enabled=True (ikke-strikt på kolonner)
# ---------------------------------------------------------------------
def test_pipeline_with_patterns_enabled_does_not_crash(capsys):
    df = make_dummy_df(30)
    _ = generate_features(df, feature_config={
        "patterns_enabled": True,  # må ikke kaste, selv hvis patterns ikke tilføjer noget
        "dropna": True
    })
    # Hvis add_all_patterns fejler, logger pipelinen en WARN via print – det er ok
    captured = capsys.readouterr()
    # ingen stram assert – blot at kørsel når hertil uden exception


# ---------------------------------------------------------------------
# Ekstra: Integration via CSV (valgfri, hurtig)
# ---------------------------------------------------------------------
def test_generate_features_from_csv_like_flow(tmp_path):
    """
    Simuler CSV-flow (uden at bruge semikolon-format) for at sikre
    at DataFrame-vej virker end-to-end.
    """
    data_path = tmp_path / "BTCUSDT_1h_test.csv"
    raw_df = make_dummy_df(32)
    raw_df.to_csv(data_path, index=False)

    df_loaded = pd.read_csv(data_path)
    features_df = generate_features(df_loaded, feature_config={"patterns_enabled": False})
    assert "ema_9" in features_df.columns
    assert not features_df.isna().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--cov=features/features_pipeline.py", "--cov-report=term-missing"])
