import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from utils.project_path import PROJECT_ROOT

# ml/train_model.py


def create_lstm_sequences(X, y, seq_length=48):
    """Omdanner data til LSTM-sekvenser (X, y)."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i : i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


def build_lstm_model(input_shape, n_classes=2):
    """Bygger en LSTM til klassifikation."""
    model = Sequential()
    model.add(LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation="softmax"))  # Klassifikation
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# AUTO PATH CONVERTED
def train_lstm_model(
    df,
    feature_cols,
    target_col="target",
    seq_length=48,
    epochs=30,
    batch_size=64,
    model_path=PROJECT_ROOT / "models" / "lstm_model.h5",
):
    """Træner og gemmer LSTM til klassifikation + feature/scaler files."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    y = df[target_col].values.astype(int)
    X_seq, y_seq = create_lstm_sequences(X_scaled, y, seq_length=seq_length)

    # Split: 80% train, 20% test (ingen shuffle, så tidsserien bevares!)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    model = build_lstm_model(
        input_shape=(seq_length, len(feature_cols)), n_classes=len(np.unique(y))
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Eval på test
    preds = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # === GEM feature-liste og scaler for downstream inference ===
    # AUTO PATH CONVERTED
    feature_path = PROJECT_ROOT / "models" / "lstm_features.csv"
    # AUTO PATH CONVERTED
    mean_path = PROJECT_ROOT / "models" / "lstm_scaler_mean.npy"
    # AUTO PATH CONVERTED
    scale_path = PROJECT_ROOT / "models" / "lstm_scaler_scale.npy"
    pd.Series(feature_cols).to_csv(feature_path, index=False, header=False)
    np.save(mean_path, scaler.mean_)
    np.save(scale_path, scaler.scale_)

    print(f"[INFO] Feature-liste gemt til {feature_path}: {feature_cols}")
    print(f"[INFO] Scaler mean og scale gemt til {mean_path} og {scale_path}")
    return model, history, scaler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Sti til feature-CSV")
    parser.add_argument("--target", type=str, default="target", help="Target-kolonne")
    parser.add_argument("--seq_length", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    # AUTO PATH CONVERTED
    parser.add_argument("--model_out", type=str, default=PROJECT_ROOT / "models" / "lstm_model.h5")
    args = parser.parse_args()

    # Indlæs data
    df = pd.read_csv(args.data, comment="#")
    df = df.dropna(subset=[args.target])

    # Robust: find feature-kolonner som i feature engineering!
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        col for col in num_cols if col not in [args.target, "future_return", "timestamp"]
    ]
    print(f"Træner på features: {feature_cols}")

    # Hvis du har tidligere feature-fil (fx fra models/lstm_features.csv), advar hvis rækkefølge er forskellig:
    # AUTO PATH CONVERTED
    feature_path = PROJECT_ROOT / "models" / "lstm_features.csv"
    if os.path.exists(feature_path):
        prev_features = pd.read_csv(feature_path, header=None)[0].tolist()
        if feature_cols != prev_features:
            print(
                f"[ADVARSEL] Dine features matcher IKKE tidligere gemt feature-liste!\nTidligere: {prev_features}\nNye: {feature_cols}"
            )
            # Evt. slet gammel model/scaler for at tvinge gen-træning

    # Træn model og gem alt nødvendigt til inference
    model, history, scaler = train_lstm_model(
        df,
        feature_cols=feature_cols,
        target_col=args.target,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_out,
    )

    print(
        f"Træning færdig. Model gemt til '{args.model_out}'. Features og scaler gemt i 'models/'."
    )
