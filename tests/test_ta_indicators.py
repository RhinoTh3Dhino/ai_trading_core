import numpy as np
import pandas as pd
import pytest

from features import ta_indicators


def _make_base_df(rows: int = 250) -> pd.DataFrame:
    """Deterministisk OHLCV-DF med monoton DatetimeIndex."""
    data = {
        "open": [1, 2, 3, 4, 5] * (rows // 5),
        "high": [2, 3, 4, 5, 6] * (rows // 5),
        "low": [1, 1, 2, 3, 4] * (rows // 5),
        "close": [1, 2, 3, 4, 5] * (rows // 5),
        "volume": [100, 200, 300, 400, 500] * (rows // 5),
    }
    df = pd.DataFrame(data)
    # brug 'h' (ikke 'H') for at undgå FutureWarning
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")
    return df


def test_add_ta_indicators_generates_expected_columns():
    df = _make_base_df()
    orig_cols = set(df.columns)
    result = ta_indicators.add_ta_indicators(df)

    # Forventede kernekolonner
    expected = [
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_200",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "rsi_28",
        "atr_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "vwap",
        "obv",
        "adx_14",
        "zscore_20",
        "supertrend",
        "volume_spike",
        "regime",
    ]
    for col in expected:
        assert col in result.columns, f"Mangler kolonne: {col}"

    # Ingen NaN efter intern dropna/udfyldning
    assert not result.isna().any().any(), "Resultat indeholder NaN"

    # Idempotens/ingen mutation af input
    assert (
        set(df.columns) == orig_cols
    ), "Input DataFrame må ikke blive muteret i kolonner"


def test_add_ta_indicators_handles_force_no_supertrend():
    df = _make_base_df()
    result = ta_indicators.add_ta_indicators(df, force_no_supertrend=True)
    # Skal stadig udstille kolonnen (fx fallback-søjle)
    assert "supertrend" in result.columns
    assert not result["supertrend"].isna().any()


def test_add_ta_indicators_handles_zero_volume_and_no_div_by_zero():
    df = _make_base_df()
    # Sæt volumen til 0 i et interval for at teste sikring mod 0-division (vwap, obv m.m.)
    df.loc[df.index[50:80], "volume"] = 0
    result = ta_indicators.add_ta_indicators(df, force_no_supertrend=True)

    # vwap og obv bør eksistere og ikke være NaN/inf
    assert "vwap" in result.columns and "obv" in result.columns
    assert np.isfinite(result["vwap"]).all(), "vwap indeholder inf/nan ved zero-volume"
    assert np.isfinite(result["obv"]).all(), "obv indeholder inf/nan ved zero-volume"


def test_add_ta_indicators_small_window_edges_dropna_effect():
    """
    Sørg for at rolling-vinduer (fx ema_200/bb_20/adx_14) håndteres og ikke ender med NaN i output.
    Hvis datasættet efter dropna er tomt, accepteres det, så længe kolonnerne findes.
    """
    df = _make_base_df(300)
    result = ta_indicators.add_ta_indicators(df, force_no_supertrend=True)

    assert result is not None
    if len(result) == 0:
        # Tomt output efter dropna er OK, men kolonner skal være til stede
        for c in ["ema_200", "bb_upper", "adx_14", "zscore_20"]:
            assert c in result.columns
    else:
        assert 0 < len(result) <= len(df)
        for c in ["ema_200", "bb_upper", "adx_14", "zscore_20"]:
            assert c in result.columns
            assert not result[c].isna().any(), f"{c} indeholder NaN"


def test_add_ta_indicators_accepts_non_datetime_index_but_returns_ok():
    """Hvis implementeringen ikke kræver DatetimeIndex eksplicit, skal funktionen stadig levere stabile features."""
    df = _make_base_df()
    df_reset = df.reset_index(drop=True)
    out = ta_indicators.add_ta_indicators(df_reset, force_no_supertrend=True)
    for c in ["ema_9", "macd", "rsi_14", "atr_14", "vwap", "obv", "regime"]:
        assert c in out.columns
    assert not out.isna().any().any()


def test_add_ta_indicators_robust_against_input_nans():
    """Indsprøjt enkelte NaN i input for at ramme eventuelle fillna/grenlogik før/during beregning."""
    df = _make_base_df()
    df.loc[df.index[3], "close"] = np.nan
    df.loc[df.index[4], "high"] = np.nan
    df.loc[df.index[5], "low"] = np.nan
    df.loc[df.index[6], "open"] = np.nan
    df.loc[df.index[7], "volume"] = np.nan

    out = ta_indicators.add_ta_indicators(df, force_no_supertrend=True)
    # Skal stadig returnere et renset sæt uden NaN
    assert not out.isna().any().any()
    # Kernefeatures skal eksistere
    for c in ["ema_21", "macd_signal", "rsi_28", "bb_middle", "vwap", "obv", "regime"]:
        assert c in out.columns


def test_add_ta_indicators_does_not_crash_with_constant_prices():
    """Konstante priser kan give 0-variance; indikatorer skal stadig være definerede (ikke NaN/inf)."""
    rows = 300
    df = pd.DataFrame(
        {
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.0),
            "low": np.full(rows, 100.0),
            "close": np.full(rows, 100.0),
            "volume": np.arange(1, rows + 1, dtype=float),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="h"),
    )
    out = ta_indicators.add_ta_indicators(df, force_no_supertrend=True)
    for c in [
        "ema_200",
        "macd",
        "rsi_14",
        "atr_14",
        "bb_upper",
        "bb_lower",
        "zscore_20",
        "adx_14",
    ]:
        assert c in out.columns
        assert not out[c].isna().any(), f"{c} indeholder NaN i konstant-marked test"
    assert np.isfinite(out["macd"]).all()


def test_add_ta_indicators_raises_on_missing_required_columns():
    """Bevidst fjern en basiskolonne for at dække raise-grenen (hvis implementeret)."""
    df = _make_base_df().drop(columns=["close"])
    with pytest.raises(Exception):
        _ = ta_indicators.add_ta_indicators(df, force_no_supertrend=True)


def test_add_ta_indicators_tiny_input_returns_empty_but_no_crash():
    """
    Ekstra branch: meget små datasæt kan resultere i tomt output efter dropna,
    men funktionen skal stadig være robust og ikke raise.
    """
    tiny = _make_base_df(50)
    out = ta_indicators.add_ta_indicators(tiny, force_no_supertrend=True)
    assert out is not None
    assert isinstance(out, pd.DataFrame)
    if len(out) == 0:
        for c in [
            "ema_21",
            "ema_200",
            "bb_upper",
            "rsi_14",
            "vwap",
            "obv",
            "supertrend",
            "regime",
        ]:
            assert c in out.columns
    else:
        assert not out.isna().any().any()
