import pandas as pd
import numpy as np

def normalize_zscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normaliserer udvalgte kolonner med Z-score og tilføjer _z kolonner."""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col + '_z'] = (df[col] - mean) / std
        else:
            df[col + '_z'] = 0
    return df

def create_lstm_sequences(df: pd.DataFrame, seq_length: int = 48):
    """Omdan DataFrame til LSTM-sekvenser (X, y) til træning."""
    X, y = [], []
    data = df.values
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
