"""
models/train_keras.py

Træner en Keras/TensorFlow model til trading-signaler (klassifikation)
- Mixed precision (kun på GPU), robust split, class weights, features, logning
- Understøtter MLflow, TensorBoard, early stopping, checkpointing
- Inkluderer automatisk feature scaling (StandardScaler)
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import platform
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Undgå TF-warnings

# === TensorFlow/Keras ===
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError("Du mangler tensorflow! Installer med: pip install tensorflow")

# === MLflow ===
try:
    import mlflow
    import mlflow.tensorflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from utils.mlflow_utils import setup_mlflow, start_mlflow_run, end_mlflow_run

    MLUTILS_AVAILABLE = True
except ImportError:
    MLUTILS_AVAILABLE = False

sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))
from utils.telegram_utils import send_message

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_keras_model.keras")
FEATURES_PATH = os.path.join(MODEL_DIR, "best_keras_features.json")
SCALER_PATH = os.path.join(MODEL_DIR, "best_keras_scaler.pkl")
LOG_PATH = os.path.join(MODEL_DIR, "train_log_keras.txt")


def log_device_status(data_path, batch_size, epochs, lr):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tf_version = tf.__version__
    python_version = platform.python_version()
    device_str = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    status_line = (
        f"{now} | TensorFlow {tf_version} | Python {python_version} | "
        f"Device: {device_str} | Data: {data_path} | Batch: {batch_size} | Epochs: {epochs} | LR: {lr}\n"
    )
    print(f"[BotStatus.md] {status_line.strip()}")
    with open("BotStatus.md", "a", encoding="utf-8") as f:
        f.write(status_line)
    try:
        send_message("🤖 " + status_line.strip())
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke sende til Telegram: {e}")


def load_csv_auto(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if str(first_line).startswith("#"):
        print("[INFO] Meta-header fundet i CSV – loader med skiprows=1")
        return pd.read_csv(file_path, skiprows=1)
    else:
        return pd.read_csv(file_path)


def build_model(input_dim, n_classes=2, hidden_dim=64, n_layers=2, dropout=0.0):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(hidden_dim, activation="relu"))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    if n_classes == 2:
        model.add(keras.layers.Dense(1, activation="sigmoid"))
    else:
        model.add(keras.layers.Dense(n_classes, activation="softmax"))
    return model


def train_keras_model(
    data_path,
    target_col="target",
    batch_size=32,
    epochs=30,
    learning_rate=1e-3,
    hidden_dim=64,
    n_layers=2,
    dropout=0.0,
    test_size=0.2,
    random_state=42,
    verbose=True,
    save_model=True,
    use_mlflow=False,
    early_stopping=True,
    patience=5,
    monitor="val_loss",
    min_delta=1e-4,
    mixed_precision=True,
    mlflow_exp="trading_ai",
):
    # === Aktiver kun mixed precision hvis der er GPU ===
    HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0
    if mixed_precision and HAS_GPU:
        try:
            from tensorflow.keras.mixed_precision import set_global_policy

            set_global_policy("mixed_float16")
            print("[INFO] Mixed precision aktiveret (float16, kun GPU)")
        except Exception as e:
            print(f"[ADVARSEL] Kunne ikke aktivere mixed precision: {e}")
    elif mixed_precision and not HAS_GPU:
        print("[ADVARSEL] Mixed precision kræver GPU – deaktiveret for CPU.")

    log_device_status(data_path, batch_size, epochs, learning_rate)

    # MLflow-setup
    if use_mlflow and not MLFLOW_AVAILABLE:
        print("❌ MLflow ikke installeret! (pip install mlflow)")
        use_mlflow = False

    tb_run_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=f"runs/{tb_run_name}", histogram_freq=1
    )

    if use_mlflow:
        if MLUTILS_AVAILABLE:
            setup_mlflow(experiment_name=mlflow_exp)
            start_mlflow_run(run_name=tb_run_name)
        else:
            mlflow.set_experiment(mlflow_exp)
            mlflow.start_run(run_name=tb_run_name)
        mlflow.log_params(
            {
                "data_path": data_path,
                "target_col": target_col,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "dropout": dropout,
                "test_size": test_size,
                "random_state": random_state,
                "early_stopping": early_stopping,
                "patience": patience,
                "monitor": monitor,
                "mixed_precision": mixed_precision and HAS_GPU,
            }
        )

    print(f"[INFO] Indlæser data fra: {data_path}")
    df = load_csv_auto(data_path)
    assert target_col in df.columns, f"target_col '{target_col}' ikke fundet!"

    drop_cols = [target_col, "timestamp"] if "timestamp" in df.columns else [target_col]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        ignored = set(X.columns) - set(X_numeric.columns)
        print(f"[ADVARSEL] Ignorerer ikke-numeriske features: {ignored}")
    X = X_numeric

    # === Feature scaling ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Gemte scaler til: {SCALER_PATH}")
        if use_mlflow:
            mlflow.log_artifact(SCALER_PATH)

    print(
        f"[INFO] Features brugt til træning: {list(X.columns)} Antal: {len(X.columns)}"
    )
    print(f"[INFO] Unikke targets: {sorted(y.unique())}")
    print(f"[INFO] Target distribution: \n{y.value_counts()}")

    if save_model:
        with open(FEATURES_PATH, "w") as f:
            json.dump(list(X.columns), f, indent=2)
        print(f"[INFO] Gemte brugte features til: {FEATURES_PATH}")
        if use_mlflow:
            mlflow.log_artifact(FEATURES_PATH)

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    if n_classes == 2:
        y_cat = y.values.astype(np.float32)
    else:
        y_cat = keras.utils.to_categorical(y, num_classes=n_classes)

    # Split uden shuffle (tidsserie)
    df = df.reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_cat[:split_idx], y_cat[split_idx:]

    print(
        "[INFO] Train slutter:",
        (
            df.iloc[split_idx - 1]["timestamp"]
            if "timestamp" in df.columns
            else split_idx - 1
        ),
    )
    print(
        "[INFO] Val starter:",
        df.iloc[split_idx]["timestamp"] if "timestamp" in df.columns else split_idx,
    )
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")
    if n_classes > 2:
        print("Target-fordeling (train):", np.sum(y_train, axis=0) / len(y_train))
        print("Target-fordeling (val):", np.sum(y_val, axis=0) / len(y_val))
    else:
        print(
            "Target-fordeling (train):\n",
            pd.Series(y_train).value_counts(normalize=True),
        )
        print(
            "Target-fordeling (val):\n", pd.Series(y_val).value_counts(normalize=True)
        )

    # Class weights (imbalance)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"[INFO] Class weights: {class_weight_dict}")

    # Byg model
    model = build_model(
        input_dim=X.shape[1],
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = "binary_crossentropy" if n_classes == 2 else "categorical_crossentropy"
    metrics = ["accuracy"]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.summary(print_fn=lambda x: print("[MODEL]", x))

    # Early stopping og checkpointing
    callbacks = [tensorboard_callback]
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                verbose=1,
                restore_best_weights=True,
            )
        )
    if save_model:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )

    # Træn
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluering
    y_pred_probs = model.predict(X_val)
    if n_classes == 2:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_val.astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)

    acc = np.mean(y_pred == y_true)
    report = classification_report(y_true, y_pred, zero_division=0)
    conf = confusion_matrix(y_true, y_pred)

    print(f"[INFO] Val accuracy: {acc:.3f}")
    print(report)
    print("Confusion matrix:\n", conf)

    log_str = (
        f"\n== Keras træningslog {datetime.now()} ==\n"
        f"Model: {MODEL_PATH}\nVal_acc: {acc:.3f}\n"
        f"Features: {list(X.columns)}\n\n"
        f"Report:\n{report}\nConfusion:\n{conf}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_str)
    print(f"[INFO] Træningslog gemt til: {LOG_PATH}")

    if use_mlflow:
        mlflow.log_metric("val_acc", acc)
        mlflow.log_artifact(LOG_PATH)
        mlflow.tensorflow.log_model(model, "model")
        if MLUTILS_AVAILABLE:
            end_mlflow_run()
        else:
            mlflow.end_run()

    try:
        send_message(f"✅ Keras træning færdig – Val acc: {acc:.3f}")
    except Exception as e:
        print(f"[ADVARSEL] Telegram fejl: {e}")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Træn Keras/TensorFlow-model til trading + MLflow, TensorBoard, mixed precision"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Sti til features-data (.csv)"
    )
    parser.add_argument(
        "--target", type=str, default="target", help="Navn på target-kolonne"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Antal epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Antal hidden units")
    parser.add_argument("--n_layers", type=int, default=2, help="Antal lag")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout-rate")
    parser.add_argument(
        "--mlflow", action="store_true", help="Log til MLflow (experiment tracking)"
    )
    parser.add_argument(
        "--mlflow_exp", type=str, default="trading_ai", help="MLflow experiment name"
    )
    parser.add_argument(
        "--early_stopping", action="store_true", help="Aktiver early stopping"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (antal epoker uden forbedring)",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_accuracy"],
        help="Monitor for early stopping",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="Minimum forbedring (delta) før early stopping resetter",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Aktiver mixed precision (float16 training)",
    )
    args = parser.parse_args()

    train_keras_model(
        data_path=args.data,
        target_col=args.target,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        test_size=args.test_size,
        verbose=True,
        save_model=True,
        use_mlflow=args.mlflow,
        early_stopping=args.early_stopping,
        patience=args.patience,
        monitor=args.monitor,
        min_delta=args.min_delta,
        mixed_precision=args.mixed_precision,
        mlflow_exp=args.mlflow_exp,
    )
