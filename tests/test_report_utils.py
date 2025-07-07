import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer dine funktioner fra korrekt modul-sti
from utils.report_utils import (
    update_bot_status,
    log_to_changelog,
    print_status,
    build_telegram_summary,
    backup_file,
    export_trade_journal
)

# Sikrer at 'outputs/' eksisterer (ellers fejler test)
os.makedirs("outputs", exist_ok=True)

# Opret test-csv med portefølje-metrics, hvis den ikke findes
test_csv_path = "outputs/portfolio_metrics_latest.csv"
if not os.path.exists(test_csv_path):
    with open(test_csv_path, "w", encoding="utf-8") as f:
        f.write("Navn,Balance,Profit,WinRate\nBTC,1200,12.3,0.65\nETH,900,8.9,0.54\n")

# 1. Test print_status (udskriver portefølje-metrics)
print_status(test_csv_path)

# 2. Test update_bot_status (skaber BotStatus.md)
update_bot_status(
    md_path="BotStatus.md",
    run_id="TEST001",
    portfolio_metrics_path=test_csv_path,
    version="vTEST",
    notes="Dette er en test.",
    plot_path=None,                 # Sæt evt. til 'outputs/dummy.png'
    trade_journal_path=None         # Sæt evt. til 'outputs/test_trade_journal.csv'
)

# 3. Test log_to_changelog
log_to_changelog(
    run_id="TEST001",
    version="vTEST",
    notes="Første changelog-test.",
    changelog_path="CHANGELOG.md"
)

# 4. Test build_telegram_summary
telegram_msg = build_telegram_summary(
    run_id="TEST001",
    portfolio_metrics_path=test_csv_path,
    version="vTEST",
    extra_msg="Ekstra status fra test."
)
print("--- Telegram-summary ---")
print(telegram_msg)

# 5. Test backup_file
backup_file(test_csv_path)

# 6. Test export_trade_journal
df = pd.DataFrame([
    {"tid": "2024-07-07", "symbol": "BTC", "ret": 0.05},
    {"tid": "2024-07-07", "symbol": "ETH", "ret": 0.02}
])
export_trade_journal(df, "outputs/test_trade_journal.csv")
