# tests/test_preprocessing_more.py
import numpy as np
import pandas as pd
import pytest
from features import preprocessing as prep


def test_normalize_zscore_std_zero_and_missing_col():
    # 'a' har std=0 -> _z bliver 0; 'missing' ignoreres; 'b' får normal Z-score
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1.0, 2.0, 3.0]})
    out = prep.normalize_zscore(df, ["a", "b", "missing"])
    assert "a_z" in out and out["a_z"].eq(0).all()
    assert "b_z" in out and abs(out["b_z"].mean()) < 1e-9  # ca. 0-mean


def test_clean_dataframe_outlier_clip_and_normalize_true():
    # indeholder inf + grove outliers; vi beder også om normalize=True grenen
    df = pd.DataFrame({"x": [0, 0, 10, 0, 0], "y": [1, 1, 1, 1, 100], "z": [np.inf, 2, 3, 4, 5]})
    out = prep.clean_dataframe(df, features=["x", "y", "z"], outlier_z=3.0, dropna=True, normalize=True)
    # inf-rækken og outliers bør være væk og kolonnerne skalerede
    assert len(out) < len(df)
    assert {"x", "y", "z"}.issubset(out.columns)
    # skaleret (ikke kæmpe værdier)
    assert out["x"].abs().max() < 5
    assert out["y"].abs().mean() < 5


def test_clean_dataframe_dropna_false_and_no_numeric_features():
    # (a) dropna=False grenen: NaN bliver ikke droppet før outlier-masken
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    out = prep.clean_dataframe(df, features=["x"], dropna=False, normalize=False)
    assert len(out) <= len(df)  # må gerne droppe via z-masken, men grenen er ramt

    # (b) ingen numeriske features -> "Ingen numeriske..."-gren
    df2 = pd.DataFrame({"s": ["a", "b", "c"]})
    out2 = prep.clean_dataframe(df2, features=[], dropna=False, normalize=False)
    assert out2.equals(df2)


def test_create_lstm_sequences_shapes_and_prepare_ml_data():
    # create_lstm_sequences rammes
    df = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
    X, y = prep.create_lstm_sequences(df, seq_length=2)
    assert X.shape == (3, 2, 1) and y.shape == (3, 1)

    # prepare_ml_data happy path + fejlgren for manglende features
    df2 = pd.DataFrame({"close": [10, 11, 10, 12], "f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
    out = prep.prepare_ml_data(df2, ["f1", "f2"], target_col="close", target_shift=-1)
    assert set(out.columns) == {"f1", "f2", "target"}
    # pct_change(1) + shift(1) giver to NaN i target -> to rækker droppes
    assert len(out) == len(df2) - 2
    # konkret værditjek af target (næste periodes pct-ændring)
    np.testing.assert_allclose(out["target"].to_numpy(), [0.1, -0.090909], rtol=1e-6, atol=1e-9)

    with pytest.raises(ValueError):
        prep.prepare_ml_data(df2, ["missing"], target_col="close", target_shift=-1)
