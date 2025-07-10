import subprocess
import sys
import glob
import os
import argparse

# === Tilføj projektroden til sys.path for robuste imports (fungerer både CLI og VS Code) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON = sys.executable

# === Argumenter til CLI ===
parser = argparse.ArgumentParser(description="Kør hele AI Trading Bot pipeline")
parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol (fx BTCUSDT)")
parser.add_argument("--interval", type=str, default="1h", help="Tidsinterval (fx 1h, 4h)")
parser.add_argument("--lookback", type=int, default=30, help="Antal dage at hente data for")
parser.add_argument("--rolling_window", type=int, default=None, help="Antal seneste bars til retraining (valgfri)")
parser.add_argument("--feature_version", type=str, default="v1.0.0", help="Feature-version tag")
args = parser.parse_args()

SYMBOL = args.symbol
INTERVAL = args.interval
LOOKBACK = args.lookback
ROLLING_WINDOW = args.rolling_window
FEATURE_VERSION = args.feature_version

def run_command(cmd_list, step_name):
    print(f"\n🔹 {step_name} ...")
    # Sæt PYTHONPATH for child-process (arver også evt. andre env vars)
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    try:
        subprocess.run(cmd_list, check=True, env=env)
        print(f"✅ {step_name} færdig!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} fejlede: {e}")
        print(f"❌ Kommando: {' '.join(cmd_list)}")
        sys.exit(1)

def main():
    # Step 1: Hent rådata
    fetch_cmd = [
        PYTHON, "fetch_data/fetch_binance_data.py",
        "--symbol", SYMBOL,
        "--interval", INTERVAL,
        "--outdir", "data",
        "--lookback", str(LOOKBACK)
    ]
    if ROLLING_WINDOW:
        fetch_cmd += ["--rolling_window", str(ROLLING_WINDOW)]

    run_command(fetch_cmd, f"Hent rådata for {SYMBOL} {INTERVAL}")

    # Find nyeste fil
    datafiles = sorted(glob.glob(f"data/{SYMBOL}_{INTERVAL}_*.csv"))
    if not datafiles:
        print("❌ Ingen datafiler fundet efter download. Stopper.")
        sys.exit(1)
    input_file = datafiles[-1]
    print(f"✅ Nyeste datafil: {input_file}")

    # Step 2: Feature engineering
    feature_dir = "outputs/feature_data"
    os.makedirs(feature_dir, exist_ok=True)
    feature_output = f"{feature_dir}/{SYMBOL.lower()}_{INTERVAL}_features_{FEATURE_VERSION}.csv"
    feature_cmd = [
        PYTHON, "features/feature_engineering.py",
        "--input", input_file,
        "--output", feature_output,
        "--version", FEATURE_VERSION
    ]
    run_command(feature_cmd, "Feature engineering")

    # Step 3: Kør engine (strategi, backtest, ML-signaler)
    engine_cmd = [
        PYTHON, "bot/engine.py",
        "--features", feature_output,
        "--symbol", SYMBOL,
        "--interval", INTERVAL
    ]
    run_command(engine_cmd, "Kør engine")

    print("\n🎉 Pipeline kørt færdig uden fejl!")

if __name__ == "__main__":
    main()
