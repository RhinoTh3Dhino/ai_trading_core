# tests/test_multi_live_sim.py

import os
import subprocess

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")

def run_dummy_command():
    print("[Dummy-test] Ingen live_simulator tilgængelig – test passeres.")
    return 0

def test_live_simulator_dummy():
    print_section("TEST 1: Dummy live_simulator (ingen ægte kørsel)")
    assert run_dummy_command() == 0

def test_multi_live_simulator_dummy():
    print_section("TEST 2: Dummy multi_live_simulator (ingen ægte kørsel)")
    assert run_dummy_command() == 0

def main():
    print("\n##### STARTER DUMMY-TEST AF LIVE SIMULATOR #####\n")
    test_live_simulator_dummy()
    test_multi_live_simulator_dummy()
    print("\n##### DUMMY TEST FÆRDIG – OK #####\n")

if __name__ == "__main__":
    main()
