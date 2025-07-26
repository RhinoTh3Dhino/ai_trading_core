from utils.project_path import PROJECT_ROOT
from pathlib import Path
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

print("=== TEST: START af NY version af test_features_pipeline.py ===")

# Sørg for at working dir er projektroden (så relative paths virker!)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import features pipeline
try:
    from features.features_pipeline import generate_features, save_features, load_features
except Exception as e:
    print("[FEJL] Kunne ikke importere features_pipeline:", e)
    sys.exit(1)

# Brug kun Path-objekter!
DEFAULT_DATA_PATH = str(Path(PROJECT_ROOT) / "data" / "test_data/BTCUSDT_1h_test.csv")
DEFAULT_SYMBOL = "BTC"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_VERSION = "test"

def parse_args():
    parser = argparse.ArgumentParser(description="Test features pipeline")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path til testdata (CSV)")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Symbol, fx BTC")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME, help="Timeframe, fx 1h")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION, help="Version-label, fx test")
    return parser.parse_args()

def make_version_with_timestamp(version):
    ts = datetime.now().strftime("%Y%m%d")
    return f"{version}_{ts}"

def ensure_dir_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class DummyArgs:
    data_path = DEFAULT_DATA_PATH
    symbol = DEFAULT_SYMBOL
    timeframe = DEFAULT_TIMEFRAME
    version = DEFAULT_VERSION

def test_generate_features_pipeline():
    args = DummyArgs()
    print(f"[INFO] Kører med data_path={args.data_path}")

    # Opret output-folder med Path (robust til både str og Path)
    ensure_dir_exists(Path(PROJECT_ROOT) / "outputs" / "feature_data")

    assert os.path.exists(args.data_path), f"Testdata mangler: {args.data_path}"
    raw_df = pd.read_csv(args.data_path, sep=";")
    assert len(raw_df) > 0, "Rådata er tom"

    if "datetime" in raw_df.columns:
        raw_df.rename(columns={"datetime": "timestamp"}, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

    min_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in min_cols:
        assert col in raw_df.columns, f"Kolonne mangler: {col}"

    features_df = generate_features(raw_df)
    assert not features_df.isnull().values.any(), "Features indeholder NaN!"

    expected_cols = [
        "rsi_14", "rsi_28", "macd", "macd_signal", "ema_9", "ema_21", "ema_50",
        "atr_14", "vwap", "bb_upper", "bb_lower", "return", "pv_ratio",
        "volume_spike", "regime"
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Feature mangler: {col}"

    if "target" in features_df.columns:
        assert not features_df["target"].isnull().any(), "Target indeholder NaN!"
    else:
        print("[INFO] Ingen target-kolonne fundet (ikke nødvendigt for denne test)")

    version_ts = make_version_with_timestamp(args.version)
    path = save_features(features_df, args.symbol, args.timeframe, version_ts)
    assert os.path.exists(path), f"Featurefil blev ikke gemt: {path}"

    loaded_df = load_features(args.symbol, args.timeframe, version_prefix=version_ts)
    assert len(loaded_df) > 0, "Indlæst features er tom"

    print("✅ Alle feature-tests bestået!")

if __name__ == "__main__":
    args = parse_args()
    DummyArgs.data_path = args.data_path
    DummyArgs.symbol = args.symbol
    DummyArgs.timeframe = args.timeframe
    DummyArgs.version = args.version
    test_generate_features_pipeline()
