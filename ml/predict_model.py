# ml/predict_model.py

import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def create_lstm_sequences(df: pd.DataFrame, seq_length: int = 48):
    """
    Omdanner DataFrame til LSTM sekvenser (X) til prediction.
    Returnerer numpy array af shape (n_samples, seq_length, n_features).
    """
    data = df.values
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)


def load_trained_model(model_path: str):
    """
    Loader en Keras model fra disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model-fil ikke fundet: {model_path}")
    model = load_model(model_path)
    return model


def predict_with_model(df: pd.DataFrame, model_path: str, seq_length: int = 48) -> np.ndarray:
    """
    Forbereder data og laver prediktioner med den trænede model.
    Args:
        df: DataFrame med feature-data i samme rækkefølge og form som under træning.
        model_path: Sti til den trænede model (.h5).
        seq_length: Sekvenslængde som modellen forventer.
    Returns:
        np.ndarray med modelprediktioner (fx sandsynligheder, scores eller classes).
    """
    model = load_trained_model(model_path)
    X = create_lstm_sequences(df, seq_length)
    preds = model.predict(X)
    return preds


if __name__ == "__main__":
    # Test eksempel: indlæs features, kald predict og print resultater
    import sys
    from datetime import datetime

    if len(sys.argv) < 3:
        print("Brug: python predict_model.py <feature_csv_path> <model_path>")
        sys.exit(1)

    feature_csv = sys.argv[1]
    model_path = sys.argv[2]

    df = pd.read_csv(feature_csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop(columns=["timestamp"])  # Fjern timestamp for ML input

    print(f"Loader model fra {model_path}...")
    predictions = predict_with_model(df, model_path)

    print(f"Prediktioner shape: {predictions.shape}")
    print("Seneste 5 prediktioner:")
    print(predictions[-5:])
