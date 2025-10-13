"""
trainers/train_lightgbm_mlflow.py

Version af LightGBM træner med MLflow logging aktiveret.
Bruges til eksperiment-tracking og model versionering.
"""

from utils.project_path import ensure_project_root

ensure_project_root()

import argparse
import os
import sys
from datetime import datetime

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from utils.telegram_utils import send_message


def train_with_mlflow(data_path, target_col, n_estimators, learning_rate, experiment):
    df = pd.read_csv(data_path)
    assert target_col in df.columns, "Target kolonne mangler"

    X = df.drop(columns=[target_col, "timestamp"], errors="ignore").select_dtypes(
        include=[np.number]
    )
    y = df[target_col]

    split_idx = int(0.8 * len(df))
    X_train, X_val, y_train, y_val = (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        mlflow.log_metric("val_accuracy", acc)

        print(f"✅ MLflow run færdig – Accuracy: {acc:.3f}")
        mlflow.lightgbm.log_model(model, "model")

        try:
            send_message(f"✅ MLflow træning færdig – Val acc: {acc:.3f}")
        except Exception as e:
            print(f"[ADVARSEL] Telegram fejl: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--experiment", type=str, default="lightgbm_exp")
    args = parser.parse_args()

    train_with_mlflow(
        args.data, args.target, args.n_estimators, args.learning_rate, args.experiment
    )
