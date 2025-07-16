"""
tests/test_pipeline.py

Test-script til automatisk validering af hele ML/DL pipeline
for AI trading bot – nu med ensemble voting!
Kører både ML (XGBoost/sklearn) og DL (PyTorch)
på dummy-data eller din egen dataset – og tjekker at ALTING virker.

Funktioner:
- Automatisk data split (train/val/test) på tidsrække
- ML-model: træning, evaluering, gem/indlæsning, metrics
- DL-model (PyTorch): træning på GPU, early stopping, gem/indlæs, metrics
- Ensemble voting: kombinerer ML og DL signaler
- Print af ALLE nøgletal, warnings, fejl
- Automatisk backup af bedste modeller
- Kode er 100% klar til VS Code/GitHub Actions

Brug:
$ python -m tests.test_pipeline

Kræver: torch, scikit-learn, joblib, numpy, pandas
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# === NYT: Import af avancerede metrics ===
from utils.metrics_utils import advanced_performance_metrics

# ======= KONFIGURATION =======
FEATURE_COLS = ["ATR", "RSI", "EMA", "MACD", "price"]
TARGET_COL = "target"
BATCH_SIZE = 256
EPOCHS = 20
PATIENCE_LIMIT = 5
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ======= ENSEMBLE VOTING =======
def ensemble_voting(*model_preds):
    preds = np.vstack(model_preds)
    voted = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
    return voted

# ======= DATA SPLIT =======
def split_data(df):
    n = len(df)
    train = df.iloc[:int(n*0.7)]
    val   = df.iloc[int(n*0.7):int(n*0.9)]
    test  = df.iloc[int(n*0.9):]
    log(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    assert train.index.max() < val.index.min() < test.index.min(), "Datasplit overlappede!"
    return train, val, test

# ======= ML: RandomForest =======
def train_ml_model(train, val, test):
    log("Træner ML-model (RandomForest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    joblib.dump(model, "models/best_ml_model.pkl")
    metrics = {}
    all_preds = {}
    for name, data in zip(['Train','Val','Test'], [train, val, test]):
        preds = model.predict(data[FEATURE_COLS])
        acc = accuracy_score(data[TARGET_COL], preds)
        # NYT: Udregn avanceret metrics for test
        if name == "Test":
            perf_metrics = advanced_performance_metrics(
                pd.DataFrame({
                    "profit": np.where(preds==data[TARGET_COL], 1, -1)  # Simpel: 1 for korrekt, -1 for forkert
                }),
                pd.DataFrame({
                    "balance": np.cumsum(np.where(preds==data[TARGET_COL], 1, -1)) + 1000
                }),
                initial_balance=1000
            )
            log(f"Avanceret test-metrics: {perf_metrics}")
        metrics[name] = acc
        all_preds[name] = preds
        log(f"{name}-accuracy: {acc:.4f}")
        log(f"{name} report:\n{classification_report(data[TARGET_COL], preds)}")
    model2 = joblib.load("models/best_ml_model.pkl")
    test_pred_reload = model2.predict(test[FEATURE_COLS])
    assert np.array_equal(test_pred_reload, model.predict(test[FEATURE_COLS])), "Model gem/indlæs fejl!"
    return model, metrics, all_preds

# ======= DL: PyTorch netværk =======
class TradingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_dl_model(train, val, test):
    log("Træner DL-model (PyTorch)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    X_train = torch.tensor(train[FEATURE_COLS].values, dtype=torch.float32)
    y_train = torch.tensor(train[TARGET_COL].values, dtype=torch.long)
    X_val   = torch.tensor(val[FEATURE_COLS].values, dtype=torch.float32)
    y_val   = torch.tensor(val[TARGET_COL].values, dtype=torch.long)
    X_test  = torch.tensor(test[FEATURE_COLS].values, dtype=torch.float32)
    y_test  = torch.tensor(test[TARGET_COL].values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    model = TradingNet(len(FEATURE_COLS), len(np.unique(train[TARGET_COL]))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device))
            val_loss = loss_fn(val_preds, y_val.to(device)).item()
        log(f"Epoch {epoch+1:02d}: train_loss={total_loss/len(train_loader):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), "models/best_dl_model.pt")
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                log("Early stopping aktiveret!")
                break

    model.load_state_dict(torch.load("models/best_dl_model.pt"))
    model.eval()
    all_preds = {}
    with torch.no_grad():
        for name, data, X, y in zip(['Train','Val','Test'],
                                   [train, val, test],
                                   [X_train, X_val, X_test],
                                   [y_train, y_val, y_test]):
            preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y.cpu().numpy(), preds)
            # NYT: Avanceret test-metrics for Test
            if name == "Test":
                perf_metrics = advanced_performance_metrics(
                    pd.DataFrame({
                        "profit": np.where(preds==y.cpu().numpy(), 1, -1)
                    }),
                    pd.DataFrame({
                        "balance": np.cumsum(np.where(preds==y.cpu().numpy(), 1, -1)) + 1000
                    }),
                    initial_balance=1000
                )
                log(f"Avanceret test-metrics (DL): {perf_metrics}")
            all_preds[name] = preds
            log(f"{name} accuracy (DL): {acc:.4f}")
            log(f"{name} report (DL):\n" + classification_report(y.cpu().numpy(), preds))
    return model, all_preds

# ======= PIPELINE =======
def run_pipeline(df):
    log("=== TEST AF ML/DL PIPELINE + ENSEMBLE START ===")
    train, val, test = split_data(df)
    ml_model, ml_metrics, ml_preds = train_ml_model(train, val, test)
    dl_model, dl_preds = train_dl_model(train, val, test)
    # ======= ENSEMBLE VOTING PÅ TEST =======
    log("Kører ensemble voting (ML + DL)...")
    ens_test = ensemble_voting(ml_preds['Test'], dl_preds['Test'])
    ens_acc = accuracy_score(test[TARGET_COL], ens_test)
    # NYT: Avanceret metrics for ensemble
    perf_metrics = advanced_performance_metrics(
        pd.DataFrame({
            "profit": np.where(ens_test==test[TARGET_COL].values, 1, -1)
        }),
        pd.DataFrame({
            "balance": np.cumsum(np.where(ens_test==test[TARGET_COL].values, 1, -1)) + 1000
        }),
        initial_balance=1000
    )
    log(f"Test accuracy (Ensemble): {ens_acc:.4f}")
    log(f"Avanceret ensemble-metrics: {perf_metrics}")
    log("Test report (Ensemble):\n" + classification_report(test[TARGET_COL], ens_test))
    # GEM ALLE METRICS
    out = {
        "ml_train_acc": ml_metrics["Train"],
        "ml_val_acc": ml_metrics["Val"],
        "ml_test_acc": ml_metrics["Test"],
        "dl_test_acc": accuracy_score(test[TARGET_COL], dl_preds['Test']),
        "ensemble_test_acc": ens_acc,
        **{f"ensemble_{k}": v for k, v in perf_metrics.items()},
        "datetime": datetime.now().isoformat(),
        "feature_cols": FEATURE_COLS
    }
    pd.DataFrame([out]).to_csv("outputs/test_pipeline_metrics.csv", index=False)
    # GEM PREDICTIONS FOR VISUALISERING/DEBUG
    pred_df = pd.DataFrame({
        "ml_pred": ml_preds['Test'],
        "dl_pred": dl_preds['Test'],
        "ensemble_pred": ens_test,
        "target": test[TARGET_COL].values
    })
    pred_df.to_csv("outputs/test_pipeline_predictions.csv", index=False)
    log("Pipeline gennemført UDEN fejl. Alle modeller og metrics gemt.")
    log("Tjek outputs/ og models/ for resultater, predictions og modeller!")

# ======= MAIN =======
if __name__ == "__main__":
    # Dummy testdata til CI eller udvikling:
    df = pd.DataFrame({
        "ATR": np.random.rand(1000),
        "RSI": np.random.rand(1000),
        "EMA": np.random.rand(1000),
        "MACD": np.random.rand(1000),
        "price": np.random.rand(1000),
        "target": np.random.randint(0, 2, 1000)
    })
    df.index = pd.RangeIndex(len(df))  # Sikrer unik index til split
    run_pipeline(df)
