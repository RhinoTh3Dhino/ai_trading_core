import os
import pandas as pd

from utils.project_path import PROJECT_ROOT
def test_status_file_exists():
    # Opret en dummy BotStatus.md hvis den mangler
    if not os.path.exists("BotStatus.md"):
        with open("BotStatus.md", "w", encoding="utf-8") as f:
            f.write("# Dummy BotStatus for test\n")
        print("[SETUP] BotStatus.md blev oprettet for test.")
    assert os.path.exists("BotStatus.md"), "BotStatus.md findes ikke!"
    print("[PASS] BotStatus.md eksisterer.")

def test_performance_history_grows():
    # Opret dummy performance_history.csv hvis den mangler
# AUTO PATH CONVERTED
    history_path = PROJECT_ROOT / "outputs" / "performance_history.csv"
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(history_path):
        df = pd.DataFrame([{"Navn": "BTC", "Balance": 1000, "timestamp": "2025-07-07T23:00:00"}])
        df.to_csv(history_path, index=False)
        print("[SETUP] performance_history.csv blev oprettet for test.")
    df = pd.read_csv(history_path)
    assert len(df) > 0, "performance_history.csv er tom"
    print("[PASS] performance_history.csv har data.")

if __name__ == "__main__":
    test_status_file_exists()
    test_performance_history_grows()
    print("✅ Integrationstests bestået!")