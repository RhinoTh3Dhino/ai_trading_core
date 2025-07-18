# tests/test_multi_live_sim.py

import os
import subprocess
import time

def print_section(title):
    print("\n" + "="*60)
    print(title)
    print("="*60 + "\n")

def check_file_exists(path, min_size=100):
    if os.path.exists(path) and os.path.getsize(path) > min_size:
        print(f"✅ Fil OK: {path} ({os.path.getsize(path)} bytes)")
        return True
    else:
        print(f"❌ Fil mangler eller for lille: {path}")
        return False

def run_python(cmd):
    print(f"[Kører]: {cmd}")
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)
        return res
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_live_simulator_single():
    print_section("TEST 1: live_simulator.py (BTCUSDT 1h)")
    features = "outputs/feature_data/btcusdt_1h_features_v1.0_20250718.csv"
    cmd = f"python bot/live_simulator.py --features {features} --n_rows 100 --symbol btcusdt --timeframe 1h"
    res = run_python(cmd)
    assert res.returncode == 0, "live_simulator.py fejlede"
    assert check_file_exists("outputs/live_trades.csv")
    assert check_file_exists("outputs/live_balance.csv")

def test_live_simulator_fallback():
    print_section("TEST 2: live_simulator.py fallback (manglende model)")
    # Omdøb model hvis den findes
    model_path = "models/best_ml_model.pkl"
    temp_model_path = "models/best_ml_model__backup.pkl"
    if os.path.exists(model_path):
        os.rename(model_path, temp_model_path)
    features = "outputs/feature_data/btcusdt_1h_features_v1.0_20250718.csv"
    cmd = f"python bot/live_simulator.py --features {features} --n_rows 50 --symbol btcusdt --timeframe 1h"
    res = run_python(cmd)
    assert res.returncode == 0, "Fallback test fejlede"
    # Genskab model
    if os.path.exists(temp_model_path):
        os.rename(temp_model_path, model_path)

def test_multi_live_simulator_cli():
    print_section("TEST 3: multi_live_simulator.py --coins btcusdt,ethusdt")
    cmd = "python bot/multi_live_simulator.py --coins btcusdt,ethusdt --timeframes 1h --n_rows 50"
    res = run_python(cmd)
    assert res.returncode == 0, "multi_live_simulator.py fejlede"

def test_multi_live_simulator_edge():
    print_section("TEST 4: multi_live_simulator.py edge-case (forkert coin)")
    cmd = "python bot/multi_live_simulator.py --coins doesnotexist --timeframes 1h"
    res = run_python(cmd)
    # Der skal ikke fejles hårdt, men status skal printes
    assert res.returncode == 0, "multi_live_simulator edge-case fejlede"

def test_live_simulator_missing_features():
    print_section("TEST 5: live_simulator.py (forkert feature-fil)")
    cmd = "python bot/live_simulator.py --features outputs/feature_data/DOES_NOT_EXIST.csv --n_rows 10"
    res = run_python(cmd)
    # Her forventes fejlmeddelelse, men ikke crash
    assert res.returncode == 0, "live_simulator.py med manglende feature-fil crashede"

def main():
    print("\n##### STARTER AUTO-TEST AF LIVE SIMULATOR #####\n")
    test_live_simulator_single()
    test_live_simulator_fallback()
    test_multi_live_simulator_cli()
    test_multi_live_simulator_edge()
    test_live_simulator_missing_features()
    print("\n##### ALLE TESTS FÆRDIGE! #####\n")

if __name__ == "__main__":
    main()
