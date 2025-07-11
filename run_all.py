import subprocess
import sys
import glob
import os
import argparse

# === Central log-utils import ===
from utils.log_utils import log_device_status
from utils.telegram_utils import send_message

# === Tilføj projektroden til sys.path for robuste imports (CLI & VS Code) ===
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
parser.add_argument("--model_type", type=str, default="ml", choices=["ml", "dl", "ensemble"], help="Vælg model-type (ml, dl, ensemble)")
args = parser.parse_args()

SYMBOL = args.symbol
INTERVAL = args.interval
LOOKBACK = args.lookback
ROLLING_WINDOW = args.rolling_window
FEATURE_VERSION = args.feature_version
MODEL_TYPE = args.model_type

def log_to_file(line, prefix="[INFO] "):
    os.makedirs("logs", exist_ok=True)
    with open("logs/bot.log", "a", encoding="utf-8") as logf:
        logf.write(prefix + line)

def run_command(cmd_list, step_name):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_start = f"[{timestamp}] [STEP] {step_name}\n"
    log_to_file(step_start, prefix="")

    print(f"\n🔹 {step_name} ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    try:
        subprocess.run(cmd_list, check=True, env=env)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_ok = f"[{timestamp}] [OK] {step_name} færdig!\n"
        log_to_file(step_ok, prefix="")
        print(f"✅ {step_name} færdig!")
    except subprocess.CalledProcessError as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_err = f"[{timestamp}] [ERROR] {step_name} fejlede: {e}\n"
        log_to_file(step_err, prefix="")
        print(f"❌ {step_name} fejlede: {e}")
        print(f"❌ Kommando: {' '.join(cmd_list)}")
        sys.exit(1)

def main():
    # Step 0: Device-detektion, central logning og Telegram (Pro)
    device_info = log_device_status(
        context="run_all",
        extra={"symbol": SYMBOL, "interval": INTERVAL, "model_type": MODEL_TYPE},
        telegram_func=send_message,   # Logger og sender status til TG
        print_console=True
    )

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
        err = f"Ingen datafiler fundet efter download. Stopper.\n"
        log_to_file(err, prefix="[ERROR] ")
        print("❌ " + err.strip())
        sys.exit(1)
    input_file = datafiles[-1]
    ok = f"Nyeste datafil: {input_file}\n"
    log_to_file(ok)
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

    # Step 3: Kør engine (strategi, backtest, ML/DL/ensemble)
    engine_cmd = [
        PYTHON, "bot/engine.py",
        "--features", feature_output,
        "--symbol", SYMBOL,
        "--interval", INTERVAL,
        "--model_type", MODEL_TYPE,
        "--device", str(device_info.get("device_str", "cpu"))  # eller "cuda"/"cpu" hvis du vil være sikker
    ]
    run_command(engine_cmd, "Kør engine")

    print("\n🎉 Pipeline kørt færdig uden fejl!")
    log_to_file("Pipeline kørt færdig uden fejl!\n")

if __name__ == "__main__":
    main()
