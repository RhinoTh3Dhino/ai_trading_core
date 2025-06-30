# ml/train_model.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

def create_lstm_sequences(data, seq_length=48):
    """
    Omdan data til LSTM-sekvenser (X, y).
    data: numpy array eller pandas DataFrame (features).
    seq_length: antal tidssteg pr. sekvens.
    Returnerer: X, y som numpy arrays.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Forudsætter at target er i kolonne 0 (tilpas efter behov)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Bygger en simpel LSTM-model.
    input_shape: (seq_length, num_features)
    """
    model = Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))  # Regression output
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(df, features, seq_length=48, epochs=50, batch_size=64, model_path="models/lstm_model.h5"):
    """
    Træner LSTM på dataframe med features.
    df: pandas DataFrame med feature-kolonner (inkl. target som første kolonne).
    features: liste af kolonnenavne til input features.
    seq_length: antal tidssteg pr. sekvens.
    epochs: antal trænings-epoker.
    batch_size: batch size.
    model_path: sti til at gemme bedste model.
    """
    # Skaler data (fit på hele df, evt. split hvis du vil træne/test)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[features])

    # Omdan til sekvenser
    X, y = create_lstm_sequences(data_scaled, seq_length=seq_length)

    # Split train/test (fx 80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Byg model
    model = build_lstm_model(input_shape=(seq_length, len(features)))

    # Callbacks: tidlig stop og gem bedste model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]

    # Træn model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    return model, history, scaler

if __name__ == "__main__":
    # Eksempel: Load data og træn model
    import config.config as cfg

    # Indlæs feature data (fx BTCUSDT 1h)
    feature_file = f"outputs/feature_data/{cfg.COINS[0].lower()}_{cfg.TIMEFRAMES[0]}_features_v1.3_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    df = pd.read_csv(feature_file)

    # Definér features og target
    # Her antager vi 'close' som target (første kolonne)
    features = ['close'] + cfg.FEATURES['trend'] + cfg.FEATURES['momentum'] + cfg.FEATURES['volatility'] + cfg.FEATURES['volume']
    features = [f for f in features if f in df.columns]

    print(f"Træner LSTM på features: {features}")

    model, history, scaler = train_lstm_model(df, features, seq_length=48, epochs=30, batch_size=64)

    print("Træning færdig. Model gemt til 'models/lstm_model.h5'.")
